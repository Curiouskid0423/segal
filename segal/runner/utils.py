"""
Helper functions to de-clutter ActiveLearningRunner class.
"""
from mmcv.runner import BaseModule
from argparse import Namespace
from typing import List
import torch
from torchvision.transforms import InterpolationMode as IM
import torchvision.transforms.functional as TF
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

    # FIXME: Adjustment for "Pad"

    return mask

def get_heuristics_by_config(config: Namespace, sample_mode: str):

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
        elif config.active_learning.heuristic == 'sparsity':
            heuristic = get_heuristics(
                mode=sample_mode, 
                name='sparsity',
                k=hconfig.k,
                inflection=hconfig.inflection,
                alpha=1.8 if not hasattr(hconfig, 'alpha') else hconfig.alpha
            )
        else:
            raise NotImplementedError("Unknown heuristic for the provided heuristic_cfg.")
    else:
        heuristic = get_heuristics(sample_mode, config.active_learning.heuristic)

    return heuristic

def process_workflow(workflow: List, sample_rounds: int):
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