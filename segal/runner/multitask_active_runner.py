"""
Customized runner file for running MAE multitask active learning
"""
import os
import os.path as osp
import numpy as np
import time
from typing import Dict
import torch
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import get_host_info, get_dist_info
from mmcv.runner.builder import RUNNERS
from mmcv.runner import master_only
from mmcv.utils import Config
from mmseg.datasets import build_dataloader
import segal.utils.runner as utils
from segal.runner import ActiveLearningRunner
from segal.utils.method import save_reconstructed_image, get_random_crops

@RUNNERS.register_module()
class MultiTaskActiveRunner(ActiveLearningRunner):

    """
    Override ActiveLearningRunner to be compatible for multi-task learning 
    without affecting previously working code on active domain adaptation
    """

    def __init__(
        self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, meta=None, 
        max_iters=None, max_epochs=None, sample_mode=None, sample_rounds=None, warmup_only=False):

        assert batch_processor is None 
        assert sample_mode in ['pixel', 'region'], \
            "currently only pixel-sampling mode is supported for MultiTaskActiveRunner"
        
        self.warmup_only = warmup_only
        
        super(MultiTaskActiveRunner, self).__init__(
            model, batch_processor, optimizer, work_dir, logger, meta, 
            max_iters, max_epochs, sample_mode, sample_rounds)


    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        batch_size = self.data_loader.batch_size
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            # mask check before segmentation
            # utils.pixel_mask_check(
            #     data_batch, batch_size, i, self.sample_mode, logger=self.logger)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, mode='train_multitask', **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

            if hasattr(self, 'multitask_val_iter') and \
                (self.iter > 0 and self.iter % self.multitask_val_iter == 0):
                # run validation
                self.track_validation()
                # set model back to train mode
                self.mode = 'train'
                self.model.train()

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def train_mae(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            # no masking needed in MAE objective
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, mode='train_mae', **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def track_validation(self):
        """ compute validation loss at any given iteration """
        self.model.eval()
        self.mode = 'val'
        classname = self.__class__.__name__
        assert hasattr(self, 'val_dataloader'), \
            f'track_validation() is called but {classname} does not ' \
                + 'have attribute `val_dataloader`.'
        self.logger.info("tracking validation...")
        time.sleep(2) # Prevent possible deadlock during epoch transition
        self.call_hook('before_val_epoch')
        with torch.no_grad():
            for idx, data in enumerate(self.val_dataloader):
                self._inner_iter = idx
                self.call_hook('before_val_iter')
                self.run_iter(data, mode='val_multitask', train_mode=False)
                self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        self.logger.info(f"completed validation at iter {self.iter}...")
        # clear out val variables for training logs
        self.log_buffer.clear()

    def val(self, data_loader, **kwargs):
        """ default mmcv function. 
        run validation after a training epoch is complete 
        """
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, mode='val_multitask', train_mode=False)
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch')

    def query(self, data_loader, **kwargs):
        self.model.eval()
        if not self.active_sampling(sample_mode=self.sample_mode):
            self.sampling_terminate = True

    def run_iter(self, data_batch, mode: str, **kwargs):
        """
        run an iteration of the given mode. `outputs` will be losses in both 
        train modes and the `val` mode.
        """
        if not self.warmup_only and mode.startswith('train') and self.sample_mode in ['pixel', 'region']:
            ignore_index = self.sample_settings.ignore_index
            data_batch = utils.preprocess_data_and_mask(data_batch, ignore_index)

        # local variable to avoid runtime error in forward() by removing 'mask'
        data = data_batch.copy()
        data.pop('mask', None)
        
        # all 'train_mae', 'train_multitask' and 'val' are using forward_train()
        if mode == 'train_mae':
            outputs = self.model.train_step(data, 'mae', self.optimizer, **kwargs)
        elif mode == 'train_multitask':
            outputs = self.model.train_step(data, 'multitask', self.optimizer, **kwargs)
        elif mode == 'val_multitask':
            outputs = self.model.val_step(data, 'multitask', self.optimizer, **kwargs)
        else:
            module_name = self.__class__.__name__
            raise NotImplementedError(
                f" mode {mode} is not implemented for run_iter() in {module_name}")

        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() and model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        self.outputs = outputs

        if mode.startswith('train') and self.iter % 500 == 0: # and self.iter > 0:
            self.visualize_mae(num_samples=10, overlay_visible_patches=True)


    def run_one_sample_round(self, flow_per_round, sample_round, datasets, **kwargs):

        """ One `sampling round`, which consists of training and query-sampling """

        for i, flow in enumerate(flow_per_round):
            mode, epochs = flow
            assert isinstance(mode, str), f'mode in workflow must be a str, but got {type(mode)}'
            assert hasattr(self, mode), f'runner has no method named "{mode}" to run an epoch'
            
            # fall back to train_set when `mode` is not present, e.g. `query` during image-based sampling
            ds = datasets['train'] if mode.startswith('train') else datasets[mode]
            # compute dataset_size for current `mode` for logging purpose
            dataset_size = len(ds) if not (self.sample_mode=='image' and mode=='query') else len(ds.pool)
            self.logger.info(f"sample round {sample_round+1} | {mode} | epochs per round {epochs} | dataset size: {dataset_size}")
            # get `epoch_runner`
            if self.warmup_only:
                if self.iter % self.configs.mae_val_iter == 0 and self.iter:
                    epoch_runner = self.val_mae
                else:
                    epoch_runner = self.train_mae
            else:
                epoch_runner = getattr(self, mode)
            # `new_loader` is NOT used in `query` but only in [`train`, `val`]
            samples_gpu, workers_gpu =  self.configs.data.samples_per_gpu, self.configs.data.workers_per_gpu
            num_gpus = len(self.configs.gpu_ids)
            new_loader = build_dataloader(
                ds, samples_gpu, workers_gpu, num_gpus,
                dist = True if num_gpus > 1 else False,
                seed = self.configs.seed, drop_last = True, 
            ) 

            for e in range(epochs):
                epoch_runner(new_loader, **kwargs)
            
            if self.sampling_terminate: 
                return

    def run(self, datasets: Dict[str, Dataset], configs: Config = None, **kwargs):
        """
        Runs the runner. This method lives through the entire workflow and 
        calls other methods in the Runner class when appropriate.

        Args:
            datasets(dict):     dictionary of datasets. each entry is (mode, Dataset)
            configs(Config):    config file provided by user and processed with MMCV
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

        if hasattr(configs, 'multitask_validation_iter') and configs.multitask_validation_iter > 0:
            self.multitask_val_iter = configs.multitask_validation_iter
            workers =  self.configs.data.workers_per_gpu
            num_gpus = len(self.configs.gpu_ids)
            self.val_dataloader = build_dataloader(
                datasets['val'], samples_per_gpu=2, workers_per_gpu=workers, 
                num_gpus=num_gpus, dist=True if num_gpus>1 else False, 
                seed=self.configs.seed, drop_last=True)
            
        # log essential info 
        self.logger.info('start running, host: %s, work_dir: %s', get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s', self.get_hook_info())
        self.logger.info('workflow: %s', workflow)
        self.call_hook('before_run')

        # when user wants to sample regularly, duplicate the inner-workflow tuples
        workflow = utils.process_workflow(workflow, self.sample_rounds)
        if hasattr(configs, 'mae_warmup_epochs') and not self.warmup_only:
            workflow = [[('train_mae', configs.mae_warmup_epochs)]] + workflow
            
        # main loops of train-query-val
        for sample_round, flow_per_round in enumerate(workflow):
            self.logger.info(f"sample round: {sample_round}, flow: {flow_per_round}")

            self.logger.info(f"active learning sample round {sample_round+1}.")
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
                self.logger.warning("WandB dashboard display does not sync when reset_each_round=True " 
                                    + "(text log and the training system itself works fine)")

            self.run_one_sample_round(flow_per_round, sample_round, datasets, **kwargs)

        # Wait for some hooks like loggers to finish
        time.sleep(1)  
        self.call_hook('after_run')

    @master_only
    def visualize_mae(self, num_samples=8, overlay_visible_patches=False):

        self.logger.info(f"saving MAE reconstructed images...")

        # visualize the `source` during warmup and the `target` during active learning
        vis_dataset = self.query_dataset if not self.warmup_only else self.dataset
        vis_indices = [np.random.randint(0, len(vis_dataset)) for _ in range(num_samples)]
        
        save_path = osp.join(self.configs.mae_viz_dir, f'iter{self.iter}')
        crop_size = self.configs.model.backbone.img_size

        for idx in vis_indices:
            data = vis_dataset.get_raw(idx)
            img = data['img'].data.detach()
            cropped_img = get_random_crops(img, crop_size, num=1)[0] # (C, H, W)
            rec, masks = self.model.module.mae_inference(cropped_img.cuda(), return_mask=True)
            if len(rec.shape) == 4:
                rec = rec.squeeze(0)

            save_reconstructed_image(
                path=save_path, 
                ori=cropped_img, 
                rec=rec.detach(),
                img_metas=data['img_metas'],
                index=idx,
                overlay_visible_patches=overlay_visible_patches,
                masks=masks.detach(),
                patch_size=self.configs.model.backbone.patch_size
            )

