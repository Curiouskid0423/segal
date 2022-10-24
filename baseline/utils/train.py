"""
Customized functions from MMSeg
Equivalent to mmseg/apis/train.py
"""

import copy
from argparse import Namespace
import torch 
import warnings
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner
from mmcv.utils import build_from_cfg
from mmseg import digit_version
from mmseg.models import BaseSegmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger, find_latest_checkpoint
from mmseg.core import DistEvalHook, EvalHook
from baseline.runner import *
from baseline.hooks import *
# from mmcv.parallel.collate import collate as mmcv_collate_fn

def setup_runner(cfg: Namespace, model: BaseSegmentor, optimizer, logger, meta, timestamp):

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
            
    runner_cfg = copy.deepcopy(cfg.runner)
    if runner_cfg.type == 'ActiveLearningRunner':
        for mode, iteration in cfg.workflow:
            if mode == 'train':
                runner_cfg.query_epochs = iteration
            
    runner = build_runner(
        runner_cfg,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    """Set up training hooks"""
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly workaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    return runner

def setup_hooks(runner, cfg: Namespace, validate: bool, distributed: bool):
    # FIXME: make sure that ignore_index in pixel-sampling works with validation set too
    """Set up eval / validate hooks"""
    if validate:
        eval_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        eval_dataloader = build_dataloader(
            eval_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        # Switch runner
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(
            eval_hook(eval_dataloader, **eval_cfg), priority='LOW')

    """Set up user-defined hooks"""
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

def setup_model(cfg: Namespace, model: BaseSegmentor, distributed: bool):
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    return model

def setup_dataloaders(cfg: Namespace, distributed: bool, datasets):
    """
    To be compatible with EpochBasedRunner, pass in dataloaders instead of dataset 
    objects since dataloaders don't need to be reinitialized during training.
    """

    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)

    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'query', 'test', 
            'train_dataloader', 'val_dataloader', 
            'query_dataloader', 'test_dataloader'
        ]
    })
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    return [build_dataloader(ds, **train_loader_cfg) for ds in datasets]

def train_al_segmentor(
    model, datasets, cfg: Namespace, distributed=False,
    validate=False, timestamp=None, meta=None):
    
    logger = get_root_logger(cfg.log_level)

    """Put models onto GPUs"""
    model = setup_model(cfg, model, distributed)
    
    """Set up runner instance and optimizer"""
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = setup_runner(cfg, model, optimizer, logger, meta, timestamp)
    
    """Set up eval hook and custom hooks (e.g. WandB)"""
    setup_hooks(runner, cfg, validate, distributed)

    """Resume training from a checkpoint file (if given)"""
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if cfg.runner.type == 'ActiveLearningRunner':
        runner.run(datasets, configs=cfg)
    elif cfg.runner.type == 'EpochBasedRunner':
        data_loaders = setup_dataloaders(cfg, distributed, datasets)
        runner.run(data_loaders, configs=cfg, workflow=cfg.workflow)
    else:
        raise NotImplementedError(
            f"Supports for IterBasedRunner is still in development. Please use ActiveLearningRunner for the time being.")