"""
This file should perform training the way MMSeg conventionally does,
but with some Active Learning wrappers that we created.
"""

import argparse
import copy
import os
import os.path as osp
import time
import warnings
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from argparse import Namespace
from typing import Dict

""" MMCV """
import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model
from mmcv.utils import Config, DictAction, get_git_hash
""" MMSegmentation """
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes
"""Customized"""
from segal.utils.train import train_al_segmentor, flatten_nested_tuple_list as flatten, is_nested_tuple


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args

def preprocess_datasets(config: Namespace) -> Dict[str, Dataset]:

    def check_flip_before_crop(transforms):
        # edit_types = ['Resize', 'Flip', 'Crop']
        # for tf in transforms:
        #     if any([e in tf['type'] for e in edit_types]):
        #         if not ('Mask' in tf['type']):
        #             print(tf['type'])
        #             return False
        # return True
        if mode != 'train' \
            or config.runner.type in ['IterBasedRunner', 'EpochBasedRunner']:
            return True
        flip_idx, crop_idx = None, None
        for idx, tf in enumerate(transforms):
            if 'Flip' in tf['type']:
                flip_idx = idx
            elif 'Crop' in tf['type']:
                crop_idx = idx
        return flip_idx==None or crop_idx==None or flip_idx < crop_idx

    datasets = {}
    
    if is_nested_tuple(config.workflow[0]):
        assert len(config.workflow) == config.runner.sample_rounds, \
            "for irregular sampling, the number of outer tuples in `workflow` has to be equal sample_rounds"
    
    flow = flatten(config.workflow)

    assert all([mode in ['train', 'val', 'query'] for (mode, _) in flow]), \
        "workflow has to be either train, val, or query"

    for mode, _ in flow:
        data_cfg = getattr(config.data, mode)

        assert isinstance(data_cfg.pipeline, list)
        assert check_flip_before_crop(data_cfg.pipeline)

        # no need to independently create a `query` dataset in `image-sampling`
        if mode == 'query' and config.runner.sample_mode == 'image':
            continue
        # make sure to use train pipeline cuz test pipeline does not load ground truth
        if mode == 'val':
            data_cfg = copy.deepcopy(data_cfg)
            data_cfg.pipeline = config.data.train.pipeline
        # avoid creating multiple dataset for the same workflow (e.g. 'train', 'query')
        if not (mode in datasets.keys()):
            datasets[mode] = build_dataset(
                data_cfg, dict(test_mode=False if mode=='train' else True))

    return datasets

def insert_ignore_index(config: Namespace, value: int):
    assert hasattr(config, "model")
    for k in config.model.keys():
        entry = getattr(config.model, k)
        if 'loss_decode' in entry:
            entry.ignore_index = value
            entry.loss_decode.avg_non_ignore=True

def main():

    """ PART 1. Argument Passing """
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # manually insert ignore_index for pixel sampling (and region-based in future development)
    if cfg.runner.type=='ActiveLearningRunner' and cfg.runner.sample_mode != 'image':
        ignore_index = cfg.active_learning.settings.pixel.ignore_index
        insert_ignore_index(config=cfg, value=ignore_index)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # Temporarily commented out since it's taking up space.
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    # Temporarily commented out since it's taking up space.
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    """ Part 2. Build model """        
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    # Temporarily commented out since it's taking up space.
    # logger.info(model)

    """ PART 3. Dataset """
    datasets = preprocess_datasets(config=cfg)
    example_dataset = datasets['train']

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=example_dataset.CLASSES,
            PALETTE=example_dataset.PALETTE)

    # add an attribute for visualization convenience
    model.CLASSES = example_dataset.CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    
    """ PART 4. Training """
    train_al_segmentor(
        model, 
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta
        )


if __name__ == '__main__':
    main()
