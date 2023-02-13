# Copyright (c) OpenMMLab. All rights reserved.
# Extend Runner class to create ActiveLearningRunner
import os.path as osp
import platform
import shutil
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as TF
import mmcv
import numpy as np
import time
from copy import deepcopy
from typing import Union, List
from argparse import Namespace
import gc

from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel.collate import collate as mmcv_collate_fn
from mmcv.runner import BaseRunner, BaseModule, get_host_info, get_dist_info, save_checkpoint
from mmcv.runner.builder import RUNNERS
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader
from mmseg.datasets.pipelines import Compose
from segal.active.active_loop import ActiveLearningLoop
from segal.active.dataset import ActiveLearningDataset
from segal.model_wrapper import ModelWrapper
from segal.active import get_heuristics

@RUNNERS.register_module()
class ActiveLearningRunner(BaseRunner):

    """ Summary:
    A runner class created to iteratively modify dataloader to accomodate active learning 
    experiments in different settings (image, pixel, region and more). Code adpated from 
    epoch_runner.py file in MMCV codebase.
    """

    def __init__(
        self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, meta=None, max_iters=None, 
        max_epochs=None, query_epochs=None, sample_mode=None, sample_rounds=None):

        if max_iters is not None:
            raise NotImplementedError("`max_iter` argument is not supported in ActiveLearningRunner.")
        self.query_epochs = query_epochs
        self.sample_mode = sample_mode
        self.sample_rounds = sample_rounds
        self.sampling_terminate = False
        
        super().__init__(
            model, batch_processor, optimizer, work_dir, logger, meta, 
            max_iters, max_epochs=query_epochs)

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
        heuristic = get_heuristics(
            self.sample_mode, cfg.active_learning.heuristic)

        self.mask_size = query_dataset.get_raw(0)['gt_semantic_seg'][0].data.numpy().squeeze().shape
        self.get_initial_labels(settings, dataset)
        self.get_initial_labels(settings, query_dataset, dataset_type='query')

        # Train and Query dataset masks have to be consistent
        if self.sample_mode != 'image':
            dataset.masks = deepcopy(query_dataset.masks)
    
        self.active_learning_loop = ActiveLearningLoop(
            dataset = dataset, 
            query_dataset= query_dataset,
            get_probabilities = self.wrapper.predict_on_dataset, 
            heuristic = heuristic,
            configs = cfg,      
            use_cuda = torch.cuda.is_available(),
            collate_fn = mmcv_collate_fn
        )

        # Set up _max_iters based on `workflow` config
        batch_size = cfg.data.samples_per_gpu * len(cfg.gpu_ids)
        dataloader_size = len(dataset) // batch_size
        self._max_iters = self.sample_rounds * self.query_epochs * dataloader_size

    def adjust_mask(self, mask: torch.Tensor, meta: List, scale=None):
        """
        Masks are created in the size of original input (e.g. 256x512 or 1024x2048), and therefore 
        requires adjustments during training transformations such as RandomFlip or Resizing. 
        This method takes a list of meta information and edits correspondingly.

        Args:
            mask (torch.Tensor):    Mask object to be edited
            meta (List):            Meta-information containing the transformation details
            scale (bool):           A boolean indicating whether RandomResize / Scaling is applied
        """
        
        for i in range(len(mask)):
            # Adjustment for "RandomFlip"
            if 'flip' in meta[i] and meta[i]['flip']:
                axis = [1] if (meta[i]['flip_direction'] == 'horizontal') else [0]
                mask[i] = mask[i].flip(dims=axis)
            
        # Adjustment for "Resize"
        if scale != None:
            mask = TF.resize(mask.unsqueeze(1), scale, IM.NEAREST).squeeze()

        # FIXME: Add adjustment for "RandomCrop" transformation

        return mask

    def run_iter(self, data_batch, train_mode, **kwargs):
        """
        Run an iteration of training.
        
        Args:
            data_batch:         MMCV compatible data batch loaded from the dataloader
            train_mode (bool):  A boolean to indicate whether the current phase is in training mode
        """
            
        if train_mode and self.sample_mode == 'pixel':
            data_batch, mask = data_batch
            ground_truth = data_batch['gt_semantic_seg'].data[0]
            mask = self.adjust_mask(
                mask=mask, meta=data_batch['img_metas'].data[0], scale=ground_truth[0].squeeze().size())
            
            assert hasattr(self, 'cfg_al')
            ground_truth.flatten()[~mask.flatten()] = self.cfg_al.settings.pixel.ignore_index
            data_batch['gt_semantic_seg'].data[0]._data = ground_truth

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

        def pixel_mask_check(data_batch, index=0):
            true_count = np.count_nonzero(data_batch[1].numpy()) // self.data_loader.batch_size
            self.logger.info(f"Mask[{index}] check: mask's True value count = {true_count}")
            return True

        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        mask_count_var = 0  # debug variable

        for i, data_batch in enumerate(self.data_loader):
            # mask check to verify that all devices have the right number of masks
            # check every 100 batches
            if (mask_count_var < 5 and i % 100 == 0) \
                and self.sample_mode == 'pixel':
                pixel_mask_check(data_batch, i)
                mask_count_var += 1
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
        if not self.active_sampling(sample_mode=self.sample_mode):
            self.sampling_terminate = True
        
    def recursively_set_is_init(self, model: BaseModule, value: bool):
        """
        Recursively re-initialize all parameters using MMCV APIs.

        Args:
            model (BaseModule):     Model backbone to be initialized.
            value (bool):           boolean value to reset model._is_init
        """

        model._is_init = value
        for m in model.children():
            if hasattr(m, '_is_init'):
                m._is_init = value
                self.recursively_set_is_init(model=m, value=value)

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

        if sample_mode == 'image':
            self.logger.info(
                f"Epoch {self.epoch} completed. Sampled new query of size {self.sample_settings['query_size']}.")
        else:
            labelled_pix = self.active_learning_loop.num_labelled_pixels
            total_budget = self.sample_rounds * self.sample_settings.query_size
            self.logger.info(f"Epoch {self.epoch} completed. {labelled_pix} pixels labelled, total budget {total_budget}.")

        # Reset weights (backbone, neck, decode_head)
        if "DataParallel" in self.model.__class__.__name__:
            self.recursively_set_is_init(self.model.module, False)
            self.model.module.init_weights()
        else:
            self.recursively_set_is_init(self.model, False)
            self.model.init_weights()

        if hasattr(self.cfg_al, "visualize"):
            self.active_learning_loop.visualize()

        self.logger.info("Process sleep for 3 seconds.")
        time.sleep(3) # prevent deadlock before step() returns from all devices
        self.logger.info("Re-initialized weights and lr after sampling.")
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
        self.logger.info(f"Sample mode: {self.sample_mode}")
        if self.sample_mode == 'image':
            # active_set.label_randomly(sample_settings['initial_pool'])
            if dataset_type == 'train':
                active_set.label_randomly(sample_settings['initial_pool'])
            else:
                # FIXME: Label ALL images for query 
                total_size = active_set.num_labelled + active_set.num_unlabelled
                active_set.label_randomly(n=total_size)
            self.logger.info(
                f"ActiveLearningDataset created with initial pool of {sample_settings['initial_pool']}.")
        elif self.sample_mode == 'pixel':
            active_set.label_all_with_mask(mask_shape=self.mask_size)
            
        else:
            raise ValueError(
                "Unknowned sample_mode keyword. Currently supporting pixel- and image-based sample mode.")

    def create_active_sets(self, datasets: List[Dataset], configs: Namespace) -> List[Dataset]:
        """
        Converts a list of datasets (torch.utils.data.Dataset) into ActiveLearningRunner compatible datasets
        given the workflow from config object. Specifically, convert datasets in `train` and `query` stages
        into ActiveLearningDataset instances.

        Args:
            datasets(list):     list of datasets
            configs(Namespace): config file provided by user and processed with MMCV
        """
        workflow = configs.workflow
        active_sets = deepcopy(datasets)
        train_idx, query_idx = None, None
        for i, flow in enumerate(workflow):
            mode, epochs = flow 
            if mode == 'val': 
                continue

            # Instantiate both Train set and Query set
            active_sets[i] = ActiveLearningDataset(datasets[i], configs=configs)
            
            if mode == 'train':
                assert self.query_epochs % epochs == 0, \
                    f"Train epoch in `workflow` has to be a factor of `query_epochs` but got {epochs} and {self.query_epochs}."
                train_idx = i
            elif mode == 'query':
                query_idx = i

        assert train_idx!=None and query_idx!=None, "Please provide a training set in the list of `workflow`."

        self.init_active_variables(
            dataset = active_sets[train_idx],
            query_dataset = active_sets[query_idx],
            settings = self.sample_settings, 
            cfg = configs
        )

        return active_sets

    def run(self, datasets: List[Dataset], configs: Namespace = None, **kwargs):
        """
        Runs the runner. This method lives through the entire workflow and 
        calls other methods in the Runner class when appropriate.

        Args:
            datasets(list):     list of datasets
            configs(Namespace): config file provided by user and processed with MMCV
            **kwargs:           Miscellaneous arguments to pass down to epoch_runners
        """

        workflow = configs.workflow
        self.cfg_al = configs.active_learning
        self.sample_settings = getattr(self.cfg_al.settings, self.sample_mode)

        assert isinstance(datasets, list)
        assert configs != None
        assert mmcv.is_list_of(workflow, tuple)
        assert len(datasets) == len(workflow)
        
        # Set up ActiveLearningDataset instance and label the data in the specified strategy
        active_sets = self.create_active_sets(datasets, configs)
        
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info(
            'Start running, host: %s, work_dir: %s', get_host_info(), work_dir)
        self.logger.info(
            'Hooks will be executed in the following order:\n%s', self.get_hook_info())
        self.logger.info(
            'workflow: %s, query epochs per sampling round: %d epochs', workflow, self.query_epochs)
        self.call_hook('before_run')

        # Loop through the workflow until hits max_epochs
        for al_round in range(self.sample_rounds):
            self.logger.info(f"Active Learning sample round {al_round+1}.")
            self._epoch = 0

            for i, flow in enumerate(workflow):
                mode, epochs = flow

                self.logger.info(f"Current {mode} set size: {len(active_sets[i])}")
                if isinstance(mode, str):  
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError('mode in workflow must be a str, but got {}'.format(type(mode)))
                
                new_loader = build_dataloader(
                    active_sets[i],
                    configs.data.samples_per_gpu,
                    configs.data.workers_per_gpu,
                    num_gpus = len(configs.gpu_ids),
                    dist = True if len(configs.gpu_ids) > 1 else False,
                    seed = configs.seed,
                    drop_last = True
                ) 
            
                for _ in range(epochs):
                    epoch_runner(new_loader, **kwargs)

                del new_loader
                gc.collect()
                torch.cuda.empty_cache()
                
                if self.sampling_terminate:
                    return

        # Wait for some hooks like loggers to finish
        time.sleep(1)  
        self.call_hook('after_run')

    def save_checkpoint(self,
                    out_dir,
                    filename_tmpl='epoch_{}.pth',
                    save_optimizer=True,
                    meta=None,
                    create_symlink=True):
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
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
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