"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import numpy as np
import torch
import torch.utils.data as torchdata

from .heuristics import AbstractHeuristic, Random
from .dataset import ActiveLearningDataset

class ActiveLearningLoop:

    """
    A loop to synchronously update with the training loop (by unit of epoch)
    Each step will label some datapoints from the unlabeled pool, and add it 
    to the training set (labeled).

    Args:
        dataset (type ActiveLearningDataset): Dataset with some sample already labelled.
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
        get_probabilities: Callable,
        heuristic: AbstractHeuristic = Random(),
        configs: dict = {},
        max_sample=-1,
        **kwargs,
    ):
        if configs['sample_mode'] == 'image':
            self.settings = configs['image_based_settings']
        else:
            self.settings = configs['pixel_based_settings']
        self.query_size = self.settings['query_size']

        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset 
        self.max_sample = max_sample
        self.configs = configs # cfg.active_learning dictionary
        self.sample_mode = configs['sample_mode']
        assert self.sample_mode in ['pixel', 'image'], "Sample mode needs to be either pixel or image"
        self.sample_settings = configs[f'{self.sample_mode}_based_settings']

        self.kwargs = kwargs

    def step(self, pool=None) -> bool:
        """
        Sample and annotate from the pool at each step
        
        Return: 
        True if successfully stepped, False if not (thus stop traing)
        """

        # `indices` Used in torchdata.Subset
        
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
        
        # FIXME: for pixel-based sampling, you don't step thru the pool cuz every image is
        # technically available in the labelled set, but just sparsely labelled.
        if len(pool) > 0:
            uncertainty_scores = self.get_probabilities(pool, self.heuristic, **self.kwargs)
            
            if uncertainty_scores is not None:
                ranked = self.heuristic.reorder_indices(uncertainty_scores)
                if indices is not None:
                    ranked = indices[np.array(ranked)]
                if len(ranked) > 0:
                    self.dataset.label(ranked[:self.query_size])
                    return True
                    
        return False