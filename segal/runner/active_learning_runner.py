# Extend Runner class to create ActiveLearningRunner
import os.path as osp
import platform
import shutil
import torch
from torch.utils.data import Dataset
import gc
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as TF
import mmcv
import numpy as np
import time
from copy import deepcopy
from typing import List, Dict
from argparse import Namespace
# import gc # garbage collection
# from torch.profiler import profile, record_function, ProfilerActivity

from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel.collate import collate as mmcv_collate_fn
from mmcv.runner import BaseRunner, get_host_info, get_dist_info, save_checkpoint
from mmcv.runner.builder import RUNNERS
from mmseg.datasets import build_dataloader
from segal.active.active_loop import ActiveLearningLoop
from segal.active.dataset import ActiveLearningDataset
from segal.model_wrapper import ModelWrapper
from segal.runner import utils

@RUNNERS.register_module()
class ActiveLearningRunner(BaseRunner):

    """ Summary:
    A runner class created to iteratively modify dataloader to accomodate active learning 
    experiments in different settings (image, pixel, region and more). Code adpated from 
    epoch_runner.py file in MMCV codebase.
    """

    def __init__(
        self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, meta=None, max_iters=None, 
        max_epochs=None, sample_mode=None, sample_rounds=None):

        if max_iters is not None:
            raise NotImplementedError(
                f"setting `max_iters` manually is not supported in ActiveLearningRunner currently. " 
                + f"Got max_iters {max_iters}")
        assert max_epochs is not None, "`max_epochs` should not be None."
        self.sample_mode = sample_mode
        self.sample_rounds = sample_rounds
        self.sampling_terminate = False
        
        super().__init__(
            model, batch_processor, optimizer, work_dir, logger, meta, max_iters, max_epochs)

    def init_active_variables(
        self, dataset: ActiveLearningDataset, query_dataset: ActiveLearningDataset, settings: dict, cfg: Namespace):
        """
        Helper function to initialize variables for active learning experiments such as 
        ModelWrapper and ActiveLearningLoop. Also updates self._max_iters to be accurate in display.

        Args:
            dataset (ActiveLearningDataset):        Dataset with mutatable masks for training
            query_dataset (ActiveLearningDataset):  Dataset with mutatable masks for query
                                                    (Query pipeline transformations)
            settings (dict):                        Dictionary for sampling settings as specified in config file
            cfg (Namespace):                        Full config object for downstream method uses later.
        """
        
        self.dataset = dataset
        self.query_dataset = query_dataset
        self.wrapper = ModelWrapper(self.model, cfg)

        if self.sample_mode != 'image':
            self.mask_size = cfg.scale_size
            self.get_initial_labels(settings, query_dataset, dataset_type='query')
            # Train and Query dataset masks have to be consistent
            dataset.masks = deepcopy(query_dataset.masks)

        self.get_initial_labels(settings, dataset)
    
        self.active_learning_loop = ActiveLearningLoop(
            dataset = dataset, 
            query_dataset= query_dataset,
            get_probabilities = self.wrapper.predict_on_dataset, 
            heuristic = utils.get_heuristics_by_config(cfg, self.sample_mode),
            configs = cfg,      
            use_cuda = torch.cuda.is_available(),
            collate_fn = mmcv_collate_fn
        )

        self._max_iters = utils.get_max_iters(
            cfg, self.max_epochs, self.sample_mode, dataset_size=len(dataset))
        self.logger.info(f"max iters: {self.max_iters} | max epochs: {self.max_epochs}")

    def run_iter(self, data_batch, train_mode, **kwargs):
        """
        Run an iteration of training.
        
        Args:
            data_batch:         MMCV compatible data batch loaded from the dataloader
            train_mode (bool):  A boolean to indicate whether the current phase is in training mode
        """
            
        if train_mode and self.sample_mode == 'pixel':
            data_batch, mask = data_batch
            mask = mask.detach() # masks do not need gradients
            ignore_index = self.cfg_al.settings.pixel.ignore_index
            data_batch = utils.preprocess_data_and_mask(data_batch, mask, ignore_index)

        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        self.outputs = outputs

    def train(self, data_loader, **kwargs):

        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        batch_size = self.data_loader.batch_size
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            utils.pixel_mask_check(
                data_batch, batch_size, i, self.sample_mode, logger=self.logger)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def query(self, data_loader, **kwargs):
        """
        This method takes in a DataLoader with `query pipeline` and performs one step
        in ActiveLearningLoop, which should update the `mask` variable.
        """
        self.model.eval()
        if not self.active_sampling(sample_mode=self.sample_mode):
            self.sampling_terminate = True

    def reset_weights(self):
        
        if "DataParallel" in self.model.__class__.__name__:
            model = self.model.module
        else:
            model = self.model

        # config contains `resume_from`
        if hasattr(self.configs, 'load_from') and self.configs.load_from != None:
            from mmcv.runner.checkpoint import load_checkpoint
            rank, _ = get_dist_info()
            if rank == 0:
                load_checkpoint(model, filename=self.configs.load_from)
        else:
            utils.recursively_set_is_init(model, value=False)
            model.init_weights()

    def active_sampling(self, sample_mode: str):
        """
        Given the sample_mode (image or pixel), perform a round of active learning sampling, and re-initialize
        the backbone recursively before a new round of training begins.

        Args:
            sample_mode (str):  a string specifying whether `image` or `pixel` mode is desired    
        """
        
        if not self.active_learning_loop.step():
            self.logger.info("ActiveLearningLoop returns False, stopping the experiment.")
            return False 

        self.logger.info("") # skipping a line to avoid logging immediately after the progress bar
        
        if sample_mode == 'image':
            budget = self.sample_settings['budget_per_round']
            self.logger.info(f"Epoch {self.epoch} completed. Sampled new query of size {budget}.")
        else:
            labelled_pix = self.active_learning_loop.num_labelled_pixels
            budget = self.sample_settings.budget_per_round
            # total_budget = self.sample_rounds * self.sample_settings.budget_per_round
            self.logger.info(
                f"Epoch {self.epoch} completed. Labelled {labelled_pix} pixels," \
                    + f" total budget {budget} pixels x {self.sample_rounds} rounds.")

        # Reset weights (backbone, neck, decode_head)
        if not hasattr(self.cfg_al, 'reset_each_round') or self.cfg_al.reset_each_round:
            self.reset_weights()
            self.logger.info("Re-initialized weights after sampling.")
        
        # Visualize the labeled query pixels
        if hasattr(self.cfg_al, "visualize"):
            self.active_learning_loop.visualize()

        self.logger.info("Process sleep for 3 seconds.")
        time.sleep(3) # prevent deadlock before step() returns from all devices
        return True
        
    def get_initial_labels(self, sample_settings: dict, active_set: ActiveLearningDataset, dataset_type: str = 'train'):
        """
        Create initial labels randomly given the sample_mode and the ActiveLearningDataset object.
        
        Args:
            sample_mode (str):                  specifies the sampling mode. Currently supports `image` and `pixel`.
            sample_settings (dict):             a dict of settings for the specified sample_mode, specified in config file.
            active_set (ActiveLearningDataset): dataset instance to be labelled.
        """
        assert dataset_type in ['query', 'train']
        if dataset_type == 'train': # avoid printing more than once
            self.logger.info(f"sample mode: {self.sample_mode}")
        if self.sample_mode == 'image':
            if dataset_type == 'train':
                active_set.label_randomly(sample_settings['initial_pool'])
            else:
                total_size = active_set.num_labelled + active_set.num_unlabelled
                active_set.label_randomly(n=total_size)
            self.logger.info(
                f"ActiveLearningDataset created | {dataset_type} | initial pool = {sample_settings['initial_pool']}.")
        elif self.sample_mode == 'pixel':
            active_set.label_all_with_mask(mask_shape=self.mask_size, mask_type=dataset_type)
            
        else:
            raise ValueError(
                "Unknowned sample_mode keyword. Currently supporting pixel- and image-based sample mode.")

    def create_active_sets(self, datasets: Dict[str, Dataset], configs: Namespace) -> List[Dataset]:
        """
        A destructive method that converts a dictionary of datasets (torch.utils.data.Dataset) 
        into ActiveLearningRunner compatible datasets given the workflow from config object. Specifically, 
        convert datasets in `train` and `query` stages into ActiveLearningDataset instances.

        Args:
            datasets(dict):     dictionary of datasets. each entry is (mode, Dataset)
            configs(Namespace): config file provided by user and processed with MMCV
        """

        train_set_present, query_set_present = None, None

        for key in datasets.keys():
            if key == 'val' or (key == 'query' and self.sample_mode == 'image'):
                pass
            else:
                datasets[key] = ActiveLearningDataset(datasets[key], configs=configs)
            
            if key == 'train':
                train_set_present = True
            elif key == 'query':
                query_set_present = True
                
        assert train_set_present!=None and (query_set_present!=None or self.sample_mode == 'image'), \
            "Please provide a training set in the list of `workflow`."

        self.init_active_variables(
            dataset = datasets['train'], 
            query_dataset = None if self.sample_mode=='image' else datasets['query'], 
            settings=self.sample_settings, 
            cfg=configs
        )

    def run(self, datasets: Dict[str, Dataset], configs: Namespace = None, **kwargs):
        """
        Runs the runner. This method lives through the entire workflow and 
        calls other methods in the Runner class when appropriate.

        Args:
            datasets(dict):     dictionary of datasets. each entry is (mode, Dataset)
            configs(Namespace): config file provided by user and processed with MMCV
            **kwargs:           Miscellaneous arguments to pass down to epoch_runners
        """

        # assertions for type check
        assert isinstance(datasets, dict)
        assert configs != None
        assert mmcv.is_list_of(configs.workflow, tuple)

        # useful local variables
        workflow = configs.workflow
        self.configs = configs
        self.cfg_al = configs.active_learning
        self.sample_settings = getattr(self.cfg_al.settings, self.sample_mode)
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
       
        # set up ActiveLearningDataset instance and label data in the specified strategy
        self.create_active_sets(datasets, configs)
        
        # log essential info 
        self.logger.info('start running, host: %s, work_dir: %s', get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s', self.get_hook_info())
        self.logger.info('workflow: %s', workflow)
        self.call_hook('before_run')

        # when user wants to sample regularly, duplicate the inner-workflow tuples
        workflow = utils.process_workflow(workflow, self.sample_rounds)

        # main loops of train-query-val
        for sample_round, flow_per_round in enumerate(workflow):

            self.logger.info(f"Active Learning sample round {sample_round+1}.")
            reset_toggle = hasattr(self.cfg_al, "reset_each_round") and self.cfg_al.reset_each_round

            # reset learning rate
            if sample_round > 0 and reset_toggle:
                self._epoch, self._iter = 0, 0
                for m, e in flow_per_round:
                    if m == 'train':
                        self._max_epochs = e
                        self._max_iters = utils.get_max_iters(
                            configs, e, self.sample_mode, dataset_size=len(datasets['train']))
                self.logger.info("Re-initialized learning rate (lr) after sampling.")

            assert utils.check_workflow_validity(flow_per_round)

            for i, flow in enumerate(flow_per_round):
                mode, epochs = flow
                
                # fall back to train_set when `mode` is not present, e.g. `query` during image-based sampling
                ds = datasets['train'] if not (mode in datasets.keys()) else datasets[mode]
                # compute dataset_size for current `mode` for logging purpose
                dataset_size = len(ds) if not (self.sample_mode=='image' and mode=='query') else len(ds.pool)

                self.logger.info(f"sample round {sample_round} | {mode} | total epochs {epochs} | dataset size: {dataset_size}")
                if isinstance(mode, str):  
                    if not hasattr(self, mode):
                        raise ValueError(f'runner has no method named "{mode}" to run an epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError('mode in workflow must be a str, but got {}'.format(type(mode)))
                
                # `new_loader` is NOT used in `query` but only in [`train`, `val`]
                samples_gpu, workers_gpu =  configs.data.samples_per_gpu, configs.data.workers_per_gpu
                num_gpu = len(configs.gpu_ids)
                new_loader = build_dataloader(
                    ds, samples_gpu, workers_gpu, num_gpu,
                    dist = True if len(configs.gpu_ids) > 1 else False,
                    seed = configs.seed, drop_last = True, 
                    # pin_memory = False
                ) 

                for _ in range(epochs):
                    epoch_runner(new_loader, **kwargs)
                
                if self.sampling_terminate: 
                    return

        # Wait for some hooks like loggers to finish
        time.sleep(1)  
        self.call_hook('after_run')

    def save_checkpoint(
        self, out_dir, filename_tmpl='epoch_{}.pth', save_optimizer=True, meta=None, create_symlink=True):
        """Save the checkpoint.
            Args:
                out_dir (str): The directory that checkpoints are saved.
                filename_tmpl (str, optional): The checkpoint filename template,
                    which contains a placeholder for the epoch number.
                    Defaults to 'epoch_{}.pth'.
                save_optimizer (bool, optional): Whether to save the optimizer to
                    the checkpoint. Defaults to True.
                meta (dict, optional): The meta information to be saved in the
                    checkpoint. Defaults to None.
                create_symlink (bool, optional): Whether to create a symlink
                    "latest.pth" to point to the latest checkpoint.
                    Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)