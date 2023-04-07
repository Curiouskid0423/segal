"""
Miscellaneous helper functions
"""
import argparse
import pickle
import sys
import os
import os.path as osp
import warnings
from typing import Tuple

import numpy as np
import torch
import mmcv
from mmcv.runner import get_dist_info
from mmcv.utils import DictAction
from mmseg.utils import get_root_logger

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

def create_pixel_masks(
    save_path: str, dataset: torch.Tensor, mask_shape: Tuple[int, int], init_pixels: int):
    
    H, W = mask_shape
    L = len(dataset)
    rank, world = get_dist_info()
    file_of_fnames = osp.join(save_path, '_filenames_.txt')

    # only process the file names once (the first time)
    if not osp.exists(file_of_fnames) or osp.getsize(file_of_fnames) == 0:
        # assert world == 1, 'to generate the filenames for the first time, please use just 1 GPU (to be fixed)'
        write_fname_in_file(fname=file_of_fnames, dataset=dataset)
        # sys.exit("finished creating the mask filenames")

    with open(file_of_fnames, 'r+') as file_of_fnames:
        fnames = file_of_fnames.read().split(',\n')[:-1] # trim the last empty string
        if rank == 0: prog_bar = mmcv.ProgressBar(task_num=L)

        for fname in fnames:
            truncated_fname = fname.split('.')
            assert len(truncated_fname)==2, 'error during processing filenames for mask creation'
            mask_fname = truncated_fname[0] # remove the meta-filename
            full_mask_fname = osp.join(save_path, f'{mask_fname}.pkl')
            mask = np.random.permutation(H * W).reshape(W, H) < init_pixels # reshape to (W, H) to be compatible with mmcv
            if rank == 0:
                with open(full_mask_fname, 'wb') as fs:
                    pickle.dump(mask, fs)
                    prog_bar.update()

        newline_after_pbar(rank)

    return True


def write_fname_in_file(fname, dataset):
    logger = get_root_logger()
    logger.info(f"writing filenames into {fname} for mask creation (only the first time)")
    L = len(dataset)
    rank, _ = get_dist_info()
    get_fname = lambda x: x['img_metas'].data['filename'].split('/')[-1]
    file = open(fname, 'w+') 

    if rank==0: prog_bar = mmcv.ProgressBar(task_num=L)

    for i in range(L):
        name = get_fname(dataset[i])
        file.write(f"{name},\n")
        if rank == 0:
            prog_bar.update()

    file.close()


def newline_after_pbar(rank):
    """print a newline after the progress bar"""
    if rank == 0: print("\n")