"""
Customized runner file for running MAE multitask active learning
"""

import numpy as np
from torch.utils.data import Dataset
import mmcv
from mmcv.runner import get_host_info
from mmcv.runner.builder import RUNNERS
from mmseg.datasets import build_dataloader
import time
from typing import Dict
from argparse import Namespace
from segal.runner import utils
from segal.runner import ActiveLearningRunner

@RUNNERS.register_module()
class MultiTaskActiveRunner(ActiveLearningRunner):

    """
    Override ActiveLearningRunner to be compatible for multi-task learning 
    without affecting previously working code on active domain adaptation
    """

    def __init__(
        self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, 
        meta=None, max_iters=None, max_epochs=None, sample_mode=None, sample_rounds=None):

        assert batch_processor is None 
        assert sample_mode == 'pixel', \
            "currently only pixel-sampling mode is supported for MultiTaskActiveRunner"
        
        super(MultiTaskActiveRunner, self).__init__(
            model, batch_processor, optimizer, work_dir, logger, meta, 
            max_iters, max_epochs, sample_mode, sample_rounds)

    def train(self, data_loader, **kwargs):
        raise NotImplementedError(
            "plain `train()` method is not supported in MultiTaskActiveRunner, " 
            + "please specify the training mode")

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

    def train_seg(self, data_loader, **kwargs):
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
            self.run_iter(data_batch, mode='train_seg', **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.mode = 'val'
        raise NotImplementedError("val mode is not implemented yet for MultiTaskActiveRunner")

    def query(self, data_loader, **kwargs):
        self.model.eval()
        if not self.active_sampling(sample_mode=self.sample_mode):
            self.sampling_terminate = True

    def run_iter(self, data_batch, mode: str, **kwargs):
        """
        run an iteration of the given mode. `outputs` will be losses in both 
        train modes and the `val` mode.
        """
        if mode.startswith('train'):
            data_batch, mask = data_batch
            mask = mask.detach() # masks do not need gradients
            ignore_index = self.cfg_al.settings.pixel.ignore_index
            data_batch = utils.preprocess_data_and_mask(data_batch, mask, ignore_index)

        if mode == 'train_mae':
            outputs = self.model.train_step(data_batch, 'mae', self.optimizer, **kwargs)
        elif mode == 'train_seg':
            outputs = self.model.train_step(data_batch, 'seg', self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() and model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        self.outputs = outputs

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
            self.logger.info(f"sample round {sample_round} | {mode} | epochs per round {epochs} | dataset size: {dataset_size}")
            # get `epoch_runner`
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
        # self.logger.info('Hooks will be executed in the following order:\n%s', self.get_hook_info())
        self.logger.info('workflow: %s', workflow)
        self.call_hook('before_run')

        # when user wants to sample regularly, duplicate the inner-workflow tuples
        workflow = utils.process_multitask_workflow(workflow, self.sample_rounds)
        if hasattr(configs, 'mae_warmup_epochs'):
            workflow = [[('train_mae', configs.mae_warmup_epochs)]] + workflow

        # main loops of train-query-val
        for sample_round, flow_per_round in enumerate(workflow):
            self.logger.info(f"sample round: {sample_round}, flow: {flow_per_round}")

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

            self.run_one_sample_round(flow_per_round, sample_round, datasets, **kwargs)

        # Wait for some hooks like loggers to finish
        time.sleep(1)  
        self.call_hook('after_run')