"""
Customized functions from MMSeg
Equivalent to mmseg/apis/train.py
"""

import torch 
from torch.nn import CrossEntropyLoss, DataParallel
import torch.optim as optim
import warnings
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner
from mmcv.utils import build_from_cfg
from mmseg import digit_version
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger, find_latest_checkpoint
from mmseg.core import DistEvalHook, EvalHook
from baseline_dev.runner import *
from baseline_dev.hooks import *
# from mmcv.parallel.collate import collate as mmcv_collate_fn

def train_al_segmentor(model,
                    datasets,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):

    """
    - dataset (ActiveLearningDataset)
    - model (should be compatible / equivalent to nn.Module)
    """
    
    logger = get_root_logger(cfg.log_level)

    """Put models onto GPUs"""
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
        # Dev note 2022.04.07
        # - argument `model` is of type nn.Module.
        # - MMDataParallel >> nn.DataParallel
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    
    """Set up runner instance and optimizer"""
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
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

    # FIXME: make sure that ignore_index in pixel-sampling works with validation set too
    """Set up eval / validate hooks"""
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        # Switch runner
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

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

    """Resume training from a checkpoint file (if given)"""
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(datasets, configs=cfg)