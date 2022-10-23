"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import numpy as np
import torch
import torch.utils.data as torchdata
from mmseg.utils import get_root_logger
from mmcv.runner import get_dist_info
from .heuristics import AbstractHeuristic, Random
from .dataset import ActiveLearningDataset

class ActiveLearningLoop:

    """
    A loop to synchronously update with the training loop (by unit of epoch)
    Each step will label some datapoints from the unlabeled pool, and add it 
    to the training set (labeled).

    Args:
        dataset:                Type ActiveLearningDataset. Dataset with some sample already labelled.
        query_dataset:          Dataset with query_pipeline applied (same size as mask)
        get_probabilities (Function): 
            Dataset -> **kwargs -> ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic):  Heuristic from baal.active.heuristics.
        query_size (int):       Number of sample to label per step.
        max_sample (int):       Limit the number of sample used (-1 is no limit).
        **kwargs:               Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        query_dataset: torchdata.Dataset, 
        get_probabilities: Callable,
        heuristic: AbstractHeuristic = Random(),
        configs: dict = {},
        max_sample=-1,
        **kwargs,
    ):

        self.dataset = dataset 
        self.query_dataset = query_dataset
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.max_sample = max_sample
        self.configs = configs 
        self.sample_mode = configs.runner.sample_mode
        assert self.sample_mode in ['pixel', 'image'], "Sample mode needs to be either pixel or image"
        self.sample_settings = getattr(configs.active_learning.settings, self.sample_mode)
        if self.sample_mode == 'pixel':
            self.num_labelled_pixels =  self.sample_settings['initial_label_pixels']
        self.logger = get_root_logger()

        self.kwargs = kwargs

    def update_image_labelled_pool(self, dataset, indices):
        
        if len(dataset) <= 0: 
            return False

        uncertainty_scores = self.get_probabilities(dataset, self.heuristic, **self.kwargs)
        # scores_check = np.any(uncertainty_scores.astype(bool)) # check if score is np.zeros

        if uncertainty_scores is not None:
            ranked = self.heuristic.reorder_indices(uncertainty_scores)
            if indices is not None:
                ranked = indices[np.array(ranked)]
            if len(ranked) > 0:
                self.dataset.label(ranked[:self.sample_settings.query_size])
                return True

        return False

    def update_pixel_labelled_pool(self):

        query_pixel_map = self.get_probabilities(self.query_dataset, self.heuristic, **self.kwargs)
        
        # Return False if machine rank != 0
        # if not np.any(query_pixel_map): 
        #     return False
            
        query_pixel_map = query_pixel_map[:len(self.dataset.masks)]

        # Set train_set's mask to be the masks computed on query_set
        union_mask = np.logical_or(self.dataset.masks, query_pixel_map)

        # DEBUG
        # debug_indices = [0, 20, 30, 50]
        # for d in debug_indices:
        #     nz_tuple = query_pixel_map[d].nonzero()
        #     dbg_list = [ (nz_tuple[0][i], nz_tuple[1][i]) for i in range(len(nz_tuple[0]))]
        #     print(f"query_pixel_map new samples: \n{sorted(dbg_list, key=lambda x:x[0])}")
            
        #     union_tuple = union_mask[d].nonzero()
        #     dbg_list_2 = [ (union_tuple[0][i], union_tuple[1][i]) for i in range(len(union_tuple[0]))]
        #     print(f"union_tuple new samples: \n{sorted(dbg_list_2, key=lambda x:x[0])}")
        # END OF DEBUG

        rank, world = get_dist_info()
        if rank == 0:
            self.dataset.masks = union_mask
        self.num_labelled_pixels += self.sample_settings['query_size']
        
        return True

    def step(self, pool=None) -> bool:
        """
        Sample and annotate from the pool at each step
        
        Return: 
        True if successfully stepped, False if not (thus stop traing)
        """
        
        if self.sample_mode == 'image':
            pool = self.dataset.pool
            assert pool is not None, "self.dataset.pool should not be None"

            if len(pool) > 0:
                # Whether max sample size is capped
                if self.max_sample != -1 and self.max_sample < len(pool):
                    # Sample without replacement
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
                return self.update_image_labelled_pool(dataset=pool, indices=indices)

        elif self.sample_mode == 'pixel':
            return self.update_pixel_labelled_pool()

        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

        return False