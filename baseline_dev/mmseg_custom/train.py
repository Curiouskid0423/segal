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
# from mmcv.parallel.collate import collate as mmcv_collate_fn

def train_al_segmentor(model,
                    datasets,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    logger=None):

    """
    - dataset (ActiveLearningDataset)
    - model (should be compatible / equivalent to nn.Module)
    
    Plan
    - Follow active learning steps in BAAL
    """
    
    # NOTE: Runner-based implementation
    
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
        """ Dev note 2022.04.07
        - argument `model` is of type nn.Module.
        - MMDataParallel >> nn.DataParallel
        """   
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
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
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

    loader_cfg = (cfg.seed, cfg.gpu_ids, cfg.data)
    runner.run(datasets, cfg.workflow, cfg.active_learning, loader_cfg)


    # NOTE: V1 implementation from scratch
    # if logger is None:
    #     logger = get_root_logger(cfg.log_level)
    # random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    
    # if not torch.cuda.is_available():
    #     return ValueError("the experiement should only be run on gpu.")

    # active_set = ActiveLearningDataset(dataset, pool_specifics=None)
    # test_set = deepcopy(dataset)

    # active_set.label_randomly(cfg.active_learning.initial_pool)

    # heuristic = get_heuristics(
    #     cfg.active_learning['heuristic'], 
    #     cfg.active_learning['shuffle_prop'],
    #     )
    # criterion = CrossEntropyLoss()
    # model.cuda()
    # # ignoring distributed mode for now
    # model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    
    # optimizer = build_optimizer(model, cfg.optimizer)

    # model = ModelWrapper(model, criterion, logger)

    # # FIXME: `samples_per_gpu` and `workers_per_gpu` should be used to 
    # # distribute data on different GPU during training.
    # batch_size = cfg.data['samples_per_gpu'] * cfg.data['workers_per_gpu'] * len(cfg.gpu_ids)
    
    # active_loop = ActiveLearningLoop(
    #     dataset=active_set, 
    #     get_probabilities=model.predict_on_dataset,
    #     heuristic=heuristic,
    #     query_size=cfg.active_learning['query_size'],
    #     batch_size=batch_size,
    #     iterations=cfg.active_learning["iterations"],
    #     use_cuda=torch.cuda.is_available(),
    #     collate_fn = mmcv_collate_fn
    #     )


    # # Following BAAL: Reset the weights at each active learning step.
    # init_weights = deepcopy(model.state_dict())
    
    # train_epochs = cfg.workflow[0][1]

    # for epoch in range(train_epochs):

    #     # Load the initial weights.
    #     model.load_state_dict(init_weights)
    #     # Train
    #     model.train_on_dataset(
    #         active_set,
    #         optimizer,
    #         batch_size,
    #         cfg.active_learning["learning_epoch"],
    #         use_cuda=torch.cuda.is_available(),
    #         collate_fn=mmcv_collate_fn,
    #     )
    #     # Validation
    #     model.test_on_dataset(
    #         test_set, 
    #         batch_size, 
    #         use_cuda=torch.cuda.is_available(),
    #         collate_fn=mmcv_collate_fn
    #         )
    #     metrics = model.metrics
    #     # FIXME step() is buggy at `get_probabilities` function
    #     should_continue = active_loop.step()
    #     if not should_continue:
    #         break

    #     val_loss = metrics["test_loss"].value
    #     logs = {
    #         "val": val_loss,
    #         "epoch": epoch,
    #         "train": metrics["train_loss"].value,
    #         "labeled_data": active_set.labelled,
    #         "Next Training set size": len(active_set),
    #     }
    #     print(logs)