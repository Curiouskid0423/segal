"""
Customized functions from MMSeg
Equivalent to mmseg/apis/train.py
"""

import copy
from argparse import Namespace
import torch 
from torch.utils.data import Dataset
from typing import Dict, List
import warnings
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, BaseRunner, build_optimizer, build_runner
from mmcv.utils import build_from_cfg
# from mmcv.parallel.collate import collate as mmcv_collate_fn
from mmseg import digit_version
from mmseg.models import BaseSegmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger, find_latest_checkpoint
from mmseg.core import DistEvalHook, EvalHook

def is_nested_tuple(x):
    return isinstance(x, tuple) and all([isinstance(it, tuple) for it in x])

def flatten_nested_tuple_list(tuple_list: List):
    result = []
    if not isinstance(tuple_list, list):
        raise ValueError
    for tp in tuple_list:
        if is_nested_tuple(tp):
            result.extend(flatten_nested_tuple_list(list(tp)))
        else:
            result.append(tp)
    return result

def setup_runner(cfg: Namespace, model: BaseSegmentor, optimizer, logger, meta, timestamp) -> BaseRunner:
    """
    Set up a runner instance and register its training hooks.

    Args:
        cfg (NameSpace):        config file provided by user and processed with MMCV
        model (BaseSegmentor):  the segmentation backbone
        optimizer (torch.optim.Optimizer): pytorch optimizer created by build_optimizer()
        logger:                 logger instance from MMCV
        meta, timestamp:        miscellaneous arguments from MMCV default configuration
    """

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
            
    runner_cfg = copy.deepcopy(cfg.runner)

    # set max_epochs in runner's config
    if runner_cfg.type == 'ActiveLearningRunner':
        if not is_nested_tuple(cfg.workflow[0]):
            # only consider the case where `reset_each_round = False`, in the case
            # where it is False, runner.run with modify max_epochs at each round
            query_epochs = None
            for mode, iteration in cfg.workflow:
                if mode == 'train':
                    query_epochs = iteration
            assert query_epochs != None
            runner_cfg.max_epochs = runner_cfg.sample_rounds * query_epochs
        else:
            flow = flatten_nested_tuple_list(cfg.workflow)
            runner_cfg.max_epochs = sum([ep for mode, ep in flow if mode == 'train'])
    elif runner_cfg.type == 'MultiTaskActiveRunner':
        assert len(cfg.workflow) == 2, "MultiTaskActiveRunner only supports sample-regularly now"
        train_flow, query_flow = cfg.workflow
        assert query_flow[0] == 'query'
        train_base_flow, repeat = train_flow
        epochs_per_round = sum([num for el, num in train_base_flow]) * repeat
        runner_cfg.max_epochs = epochs_per_round * runner_cfg.sample_rounds
        if hasattr(cfg, 'mae_warmup_epochs'):
            runner_cfg.max_epochs += cfg.mae_warmup_epochs
            
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
    """
    Set up hooks for validation use and custom hooks (e.g. WandB).

    Args:
        runner (BaseRunner):    runner instance from MMCV, created with setup_runner() call
        cfg (NameSpace):        config file provided by user and processed with MMCV
        validate (bool):        boolean to indicate whether the hook is used in validation (default MMCV code)
        distributed (bool):     boolean to indicate whether to use distributed training
    """

    """Set up eval / validate hooks"""
    if validate:
        eval_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        eval_dataloader = build_dataloader(
            eval_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False
        )
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

def setup_model(cfg: Namespace, model: BaseSegmentor, distributed: bool) -> torch.nn.Module:
    """
    Add DataParallel wrapper around a model instance depending on whether
    using distributed training or not.

    Args:
        cfg (Namespace):        config file provided by user and processed with MMCV
        model (BaseSegmentor):  the segmentation backbone
        distributed (bool):     boolean to indicate whether to use distributed training
    """
    
    if distributed:
        
        # NOTE: set `find_unused_parameters=True` in the case of multi-task training since
        # inevitably some parameters in the separate projection heads will not be used
        if cfg.runner.type == 'MultiTaskActiveRunner':
            find_unused_parameters = True
        else:
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

def setup_dataloaders(cfg: Namespace, distributed: bool, datasets: Dict[str, Dataset]):
    """
    Create a list of dataloaders given datasets. This method is only 
    compatible with Iter / EpochBasedRunner (not ActiveLearningRunner). The 
    default MMCV code passes in dataloaders instead of dataset objects 
    since dataloaders don't need to be reinitialized during training.

    Args:
        cfg (NameSpace):        config file provided by user and processed with MMCV
        distributed (bool):     boolean to indicate whether to use distributed training
        datasets (Dict):        dictionary of datasets. each entry is (mode, Dataset)
    """

    loader_cfg = dict(
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True
    )

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
    return [build_dataloader(ds, **train_loader_cfg) for ds in datasets.values()]

def train_al_segmentor(
    model, datasets: Dict[str, Dataset], cfg: Namespace, distributed=False,
    validate=False, timestamp=None, meta=None):

    """
    Trains a Segmentor (segmentation model) in Active Learning setting given the cfg file.

    Args:
        model (BaseSegmentor):  the segmentation backbone
        datasets (Dict):        dictionary of datasets. each entry is (mode, Dataset)
        cfg (Namespace):        config file provided by user and processed with MMCV
        distributed (bool):     boolean to indicate whether to use distributed training
        validate (bool):        boolean to indicate whether the hook is used in validation (default MMCV code)
        timestamp, meta:        default MMCV arguments. 
    """
    
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

    if cfg.runner.type in ['ActiveLearningRunner', 'MultiTaskActiveRunner']:
        runner.run(datasets, configs=cfg)
    else:
        data_loaders = setup_dataloaders(cfg, distributed, datasets)
        runner.run(data_loaders, configs=cfg, workflow=cfg.workflow)