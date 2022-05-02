# Copyright (c) OpenMMLab. All rights reserved.
# Extend Runner class to create ActiveLearningRunner
import os.path as osp
import platform
import shutil
import torch
from torch.utils.data import DataLoader
import mmcv
from mmcv.parallel.collate import collate as mmcv_collate_fn
from mmcv.runner import BaseRunner, get_host_info, save_checkpoint
from mmcv.runner.builder import RUNNERS
from mmseg.datasets import build_dataloader
import time
from copy import deepcopy
from baseline_dev.active.active_loop import ActiveLearningLoop
from baseline_dev.active.dataset import ActiveLearningDataset
from baseline_dev.model_wrapper import ModelWrapper
from baseline_dev.active import get_heuristics

@RUNNERS.register_module()
class ActiveLearningRunner(BaseRunner):

    """ 
    Referenced epoch_runner.py file.
    """

    def __init__(
        self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, 
        meta=None, max_iters=None, max_epochs=None):
        super().__init__(model, batch_processor, optimizer, work_dir, logger, meta, max_iters, max_epochs)

    def init_active(self, dataset, cfg, cfg_data, gpu_ids, seed=None):
        
        self.wrapper = ModelWrapper(
            self.model, logger=self.logger, cfg={
                'data': cfg_data, 
                'al': cfg, 
                'gpu_ids': gpu_ids, 
                'seed': seed,
                }
            )
        
        batch_size = cfg_data.samples_per_gpu * len(gpu_ids)
        heuristic = get_heuristics(
            cfg['heuristic'], 
            cfg['shuffle_prop'],
            )

        self.active_learning_loop = ActiveLearningLoop(
            dataset=dataset, 
            get_probabilities=self.wrapper.predict_on_dataset, 
            heuristic=heuristic,
            query_size=cfg['query_size'],
            batch_size=batch_size,  
            use_cuda=torch.cuda.is_available(),
            collate_fn = mmcv_collate_fn
            )

    def run_iter(self, data_batch, train_mode, **kwargs):

        """ Either train, val or custom batch processing """
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
        # self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
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

    def run(self, datasets, workflow, al_cfg, loader_cfg, **kwargs):
        
        """
        datasets have to be list (of either one or two datasets typically)
        of PyTorch datasets such that ActiveLearningDataset wrapper can work.
        """

        assert isinstance(datasets, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(datasets) == len(workflow)
        
        seed, gpu_ids, cfg_data = loader_cfg

        active_sets = deepcopy(datasets)
        has_train_set = False
        for i, flow in enumerate(workflow):
            mode, _ = flow 
            if mode == 'train':
                active_sets[i] = ActiveLearningDataset(datasets[i], pool_specifics=None)
                self.init_active(
                    dataset=active_sets[i], cfg=al_cfg, cfg_data=cfg_data, gpu_ids=gpu_ids, seed=seed)
                active_sets[i].label_randomly(al_cfg.initial_pool)
                self.logger.info(f"ActiveLearningDataset created with initial pool of {al_cfg.initial_pool}.")
                has_train_set = True

        assert has_train_set is True, \
            """ Active Learning assumes training mode enabled.
                Please provide a training set."""

        for i, flow in enumerate(workflow):
            mode, _ = flow 
            if mode == 'train':
                batch_size = cfg_data.samples_per_gpu * len(gpu_ids)
                loader_len = len(active_sets[i]) // batch_size
                self._max_iters = loader_len * self._max_epochs * (self._max_epochs + 1) // 2
                break
        
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                """ for `epochs` epoch, iterate with a static dataloader """
                pool_size = -1
                for _ in range(epochs):
                    
                    """ Create loader for active learning dataset """
                    self.logger.info(f"Current {mode} set size: {len(active_sets[i])}")
                    new_loader = build_dataloader(
                        active_sets[i],
                        cfg_data.samples_per_gpu,
                        cfg_data.workers_per_gpu,
                        # cfg.gpus will be ignored if distributed
                        len(gpu_ids),
                        dist=True if len(gpu_ids) > 1 else False,
                        seed=seed,
                        drop_last=True
                    ) 
                    if mode == 'train':
                        pool_size = len(active_sets[i].pool)
                        if self.epoch >= self._max_epochs:
                            break
                    
                    epoch_runner(new_loader, **kwargs)
                    
                if mode == 'train' and (self.epoch % al_cfg['query_epoch'] == 0):
                    self.logger.info(f"Predicting on unlabeled pool. Pool size {pool_size}")
                    if not self.active_learning_loop.step():
                        cid = torch.cuda.current_device()
                        self.logger.info(f"Warning: `self.active_learning_loop.step()` returned False on cuda:{cid}")
                        continue
                    
                    self.logger.info(f"Epoch {self.epoch} completed. Sampled new query of size {al_cfg['query_size']}.")
        
        time.sleep(1)  # wait for some hooks like loggers to finish
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