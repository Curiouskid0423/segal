"""
Helper functions to de-clutter ActiveLearningRunner class.
"""
from typing import List
import numpy as np
import torch
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as TF
from mmcv.runner import BaseModule
from mmcv.utils import Config
from mmseg.utils import get_root_logger
from segal.active import get_heuristics

def recursively_set_is_init(model: BaseModule, value: bool):
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
            recursively_set_is_init(model=m, value=value)

def adjust_mask(mask: torch.Tensor, meta: List, scale=None):
    """
    Masks are created in the size of original input (e.g. 256x512 or 1024x2048), and therefore 
    requires adjustments during training transformations such as RandomFlip or Resizing. 
    This method takes a list of meta information and edits correspondingly.

    Args:
        mask (torch.Tensor):    Mask object to be edited
        meta (List):            Meta-information containing the transformation details
        scale (List):           Ground Truth scale
    """
    
    for i in range(len(mask)):
        # Adjustment for "RandomFlip"
        if 'flip' in meta[i] and meta[i]['flip']:
            axis = [1] if (meta[i]['flip_direction'] == 'horizontal') else [0]
            mask[i] = mask[i].flip(dims=axis)

        # Adjustment for "RandomCrop"
        if 'crop_bbox' in meta[i]:
            print(f"crop_box: {meta[i]['crop_bbox']}")
            crop_y1, crop_y2, crop_x1, crop_x2 = meta[i]['crop_bbox']
            mask = mask[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            
    # Adjustment for "Resize"
    if scale != None:
        mask = TF.resize(mask.unsqueeze(1), scale, IM.NEAREST).squeeze()

    return mask

def get_heuristics_by_config(config: Config, sample_mode: str):
    """
    Returns a Heuristics instance given a config instance.
    Args: 
        config (Config):    config file converted from dict to Config object
        sample_mode (str):  sample_mode will either be 'pixel', 'image' or 'region'
    Return:
        a Heuristics instance
    """
    if hasattr(config.active_learning, "heuristic_cfg"):
        hconfig = config.active_learning.heuristic_cfg
        if config.active_learning.heuristic == 'ripu':
            heuristic = get_heuristics(
                mode=sample_mode, 
                name='ripu',
                k=hconfig.k,
                use_entropy=hconfig.use_entropy,
                categories=hconfig.categories
            )
        else:
            raise NotImplementedError("Unknown heuristic for the provided heuristic_cfg.")
    else:
        heuristic = get_heuristics(sample_mode, config.active_learning.heuristic)

    return heuristic

def process_workflow(workflow: List, sample_rounds: int):
    """
    Preprocess workflow list to accomodate regular and irregular sampling.
    """
    from segal.utils.train import is_nested_tuple
    # when sample regularly
    if not is_nested_tuple(workflow[0]):
        result = [tuple(workflow) for _ in range(sample_rounds-1)]
        # trim the last round of query since it won't be used at all
        assert workflow[0][0] == 'train'
        train_epoch = workflow[0][1]
        result.append((('train', train_epoch), ))
        return result
    else:
        return workflow

def get_max_iters(
    config: Config, max_epochs, sample_mode: str, dataset_size: int):
    """
    Given the config and dataset size (required in pixel-based), compute the 
    total iterations based on the total number of epochs, considering both the 
    case of sample regularly and irregularly.
    """

    num_devices, samples_per_gpu = len(config.gpu_ids), config.data.samples_per_gpu
    effective_batch_size = samples_per_gpu * num_devices
    sample_settings = getattr(config.active_learning.settings, sample_mode)
    sample_rounds = config.runner.sample_rounds

    if sample_mode == 'pixel':
        iter_per_epoch = np.ceil(dataset_size / effective_batch_size).astype(int)
        return max_epochs * iter_per_epoch
    elif sample_mode == 'image':
        init = sample_settings.initial_pool
        query = sample_settings.budget_per_round
        total_epochs = 0
        flow = process_workflow(config.workflow, sample_rounds)
        # `iter_this_round` increases in image-based sampling over time
        for r in range(sample_rounds):
            iter_this_round = np.ceil((init + query * r) / effective_batch_size).astype(int)
            for mode, epoch in flow[r]:
                if mode == 'train':
                    total_epochs += epoch * iter_this_round
        return total_epochs
    else:
        raise NotImplementedError

def check_workflow_validity(flow_per_round):
    """
    Ensure that the workflow, whether regular or irregular sampling, 
    is valid, e.g. has no repeated arguments.
    """
    wf = [m for m, _ in flow_per_round]
    return len(wf) == 1 and wf[0] == 'train' \
        or (len(wf) == 2 and all([wf.count(k)==1 for k in ['train', 'query']])) \
        or (len(wf) == 3 and all([wf.count(k)==1 for k in ['train', 'val', 'query']]))

def pixel_mask_check(data_batch, batch_size, index, sample_mode, interval=80, logger=None):
    if index % interval == 0 and sample_mode == 'pixel':
        true_count = np.count_nonzero(data_batch['mask'].numpy()) // batch_size
        if logger == None:
            logger = get_root_logger()
        logger.info(f"Mask[{index}] check: mask's True value count = {true_count}")

def preprocess_data_and_mask(data_batch, ignore_index=255):
    """
    Given a data_batch and its corresponding masks, perform required 
    preprocessing and return the data_batch appropriately masked with ignore_index
    """
    ground_truth = data_batch['gt_semantic_seg'].data[0]
    mask = data_batch['mask'] # mask==0 means not labeled yet
    ground_truth.flatten()[~mask.flatten()] = ignore_index
    data_batch['gt_semantic_seg'].data[0]._data = ground_truth
    return data_batch