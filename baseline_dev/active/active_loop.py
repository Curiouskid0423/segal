"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import numpy as np
import torch.utils.data as torchdata
import mmcv
from mmcv.runner import get_dist_info

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
        query_size: int = 1,
        max_sample=-1,
        **kwargs,
    ):
        self.query_size = query_size
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.dataset = dataset 
        self.max_sample = max_sample
        """ 
        An ad-hoc fix to out-of-cpu-memory issue 
        Storing 500 segmentation map per time maximally
        """
        self.mem_bound = len(dataset) // 500
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
        
        if len(pool) > 0:
        
            N = len(pool) // self.mem_bound

            uncertainty_scores = []

            rank, world_size = get_dist_info()
            if rank == 0:
                prog_bar = mmcv.ProgressBar(len(pool))

            for i in range(0, len(pool), N):
                # Get logits from segmentation map; 
                # dim: (batch_size, 19 classes, img_H, img_W);
                end = i+N if i+N < len(pool) else len(pool)
                subset = torchdata.Subset(pool, range(i, end))
                partial_prob = self.get_probabilities(subset, **self.kwargs)
                if partial_prob is not None:
                    scores = self.heuristic.get_uncertainties(partial_prob)
                    uncertainty_scores.append(scores)
                
                # rank 0 worker will collect progress from all workers.
                if rank == 0:
                    for _ in range(end-i):
                        prog_bar.update()

            ranked = self.heuristic.reorder_indices(uncertainty_scores)
            if uncertainty_scores != []:
                if indices is not None:
                    # use the values in `ranked` to reorder `indices`
                    ranked = indices[np.array(ranked)]
                if len(ranked) > 0:
                    self.dataset.label(ranked[:self.query_size])
                    return True

        return False