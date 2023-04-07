"""
ActiveLearningDataset is a wrapper class around a normal 
torchdata.Dataset class to enable "iteratively adding new 
samples from the unlabeled pool".

ActiveLearningPool is a normal CustomDataset from torch 
for unlabeled dataset.
"""
from itertools import zip_longest
import torch.utils.data as torchdata
from typing import Optional, Callable, Tuple
import os
import os.path as osp
import numpy as np
from copy import deepcopy

from active.dataset.base import OracleDataset
from mmseg.datasets import build_dataset
from mmseg.utils import get_root_logger

from segal.utils.misc import create_pixel_masks

def _identity(x):
    return x

class ActiveLearningDataset(OracleDataset):
    """
    Args:
        dataset: 
            The baseline dataset, type = torchdata.Dataset
        labelled: 
            An array that acts as a mask which is greater than 1 for every data point 
            that is labelled, and 0 for every data point that is not labelled.
        make_unlabelled: 
            The function that returns an unlabelled version of a datum so that 
            it can still be used in the DataLoader.
        random_state: Set the random seed for label_randomly().
        configs: 
            Config attributes for MMCV dataset.
        last_active_steps: 
            If specified, will iterate over the last_active_steps 
            instead of the full dataset. Useful when doing partial finetuning.
    """

    def __init__(self, dataset: torchdata.Dataset, labelled: Optional[np.ndarray] = None,
        make_unlabelled: Callable = _identity, random_state=None, 
        configs: dict = None, last_active_steps: int = -1):
        
        assert configs is not None, "configs cannot be None"

        self.sample_mode = configs.runner.sample_mode
        self.settings = getattr(configs.active_learning.settings, self.sample_mode)
        self.logger = get_root_logger()

        # Initialize labelled pool to be empty
        if labelled is not None:
            self.labelled_map = labelled.astype(int)
        else:
            self.labelled_map = np.zeros(len(dataset), dtype=int)
        
        # Each entry is a DataContainer class with {'img', 'img_metas', 'gt_semantic_seg'}
        self.dataset = dataset 
        self.masks_dir = osp.join(configs.work_dir, 'masks')
        os.makedirs(self.masks_dir, exist_ok=True)

        # Reset data augmentation for the unlabelled pool (test pipeline) (image-based sampling)
        self.cfg_data = deepcopy(configs.data)
        if self.cfg_data is not None and self.sample_mode == 'image':
            self.logger.info("creating unlabelled pool dataset.")
            self.pool_dataset = build_dataset(self.cfg_data['train'])
       
        # check source-free or standard ADA
        self.source_free = True
        if hasattr(configs, "source_free") and not configs.source_free:
            self.source_free = False

        self.make_unlabelled = make_unlabelled
        self.can_label = self.check_can_label()
        
        # Constructor of OracleDataset
        super().__init__(self.labelled_map, random_state, last_active_steps)
 

    def __getitem__(self, index):
        # index should be relative to the currently available list of indices
        active_index = self.get_indices_for_active_step()[index] 
        if self.sample_mode == 'pixel':
            return self.dataset[index] # self.masks[index]
        else:
            return self.dataset[active_index]
    
    class ActiveIter:
        
        """Need an iterator class for ActiveLearningDataset"""
        def __init__(self, aldataset):
            self.index = 0
            self.aldataset = aldataset

        def __len__(self):
            return len(self.aldataset)

        def __next__(self):
            if self.index >= len(self):
                raise StopIteration

            n = self.aldataset[self.index]
            self.index += 1
            return n

    def __iter__(self):
        """Return an iterator instance"""
        return self.ActiveIter(self)

    @property
    def pool(self):
        """Returns a new Dataset made from unlabelled samples"""
        # Exclude the labelled data
        recovered_index = (~self.labelled).nonzero()[0].flatten()
        # Re-create from self.pool_dataset, which has been applied with test transform
        pool_dataset = torchdata.Subset(self.pool_dataset, list(recovered_index))
        res = ActiveLearningPool(pool_dataset, make_unlabelled=self.make_unlabelled)
        return res

    def check_can_label(self):
        if hasattr(self.dataset, "label") and callable(self.dataset.label):
            return True
        return False

    def label_all_with_mask(self, mask_shape: Tuple[int, int], mask_type: str):
        """ 
        For pixel-based sampling, all images will be placed in labelled pool initially
        but only labelled sparsely on randomly selected ones. 
        """
        # Make all images visible from the getter function by placing all images into labelled pool 
        # (FIXME: legacy from BAAL, to be removed later)
        self.label(list(range(len(self.dataset))))

        h, w = mask_shape
        N = len(self.dataset)
        init_pixels = self.settings['initial_label_pixels'] // N
        assert init_pixels < h * w, "initial_label_pixels exceeds the total number of pixels"
        assert type(init_pixels) is int, f"initial_label_pixels has to be type int but got {type(init_pixels)}"

        # create the pixel masks and save to self.masks_dir
        assert mask_type in ['train', 'query']

        if not self.source_free or mask_type == 'query':
            keep_pixels = np.prod(mask_shape) if (self.source_free and mask_type=='train') else init_pixels
            self.logger.info(f"creating masks for `{mask_type}` dataset. start with {keep_pixels} per image.")
            create_pixel_masks(
                save_path=self.masks_dir, 
                dataset=self.dataset, 
                mask_shape=mask_shape, 
                init_pixels=keep_pixels
            ) 

    def label(self, index, value=None):
        """
        Update the list of indices with heuristically selected samples.
        Index is relative to pool, not the overall OracleDataset
        Args:
            index: one or more indices to be labelled
            value: when provided, set corresponding index to the given value(s),
                   otherwise just use the original dataset's label(s)
        """
        index_list = [index] if isinstance(index, int) else index
        value_list = [value] if (value is None) or isinstance(value, int) else value
        if (value_list[0] is not None) and len(index_list) != len(value_list):
            raise ValueError("Index list and Value list does not match in ActiveLearningDataset")

        indices = self._pool_to_oracle_index(index_list)
        curr_active_step = self.curr_al_step + 1

        for index, val in zip_longest(indices, value_list, fillvalue=None):
            if not self.can_label:
                self.labelled_map[index] = curr_active_step
            elif val is not None:
                self.dataset.label(index, val)
                self.labelled_map[index] = curr_active_step
            elif val is None:
                raise ValueError("The dataset is able to label data, but no label was provided.")

    def get_raw(self, idx: int):
        """Get a datapoint from the underlying dataset."""
        return self.dataset[idx]
    
    def state_dict(self):
        """Return the state_dict, ie. the labelled map and random_state."""
        return {"labelled": self.labelled_map, "random_state": self.random_state}

    def load_state_dict(self, state_dict):
        """Load the labelled map and random_state with give state_dict."""
        self.labelled_map = state_dict["labelled"]
        self.random_state = state_dict["random_state"]


class ActiveLearningPool(torchdata.Dataset):
    """
    A dataset that represents the unlabelled pool for active learning.
    """
    def __init__(
        self, dataset: torchdata.Dataset, make_unlabelled: Callable = _identity):
        self.dataset = dataset
        self.make_unlabelled = make_unlabelled
    
    def __getitem__(self, index):
        return self.make_unlabelled(self.dataset[index])

    def __len__(self):
        return len(self.dataset)