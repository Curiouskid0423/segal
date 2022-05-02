"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import torch
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
        FIXME: An ad-hoc fix to out-of-cpu-memory issue 
        - when using 4 GPU, pred_unit = 40 ~ 60
        """
        self.pred_unit = 50
        self.kwargs = kwargs

    def step(self, pool=None) -> bool:
        """
        Sample and annotate from the pool at each step
        
        Return: 
        True if successfully stepped, False if not (thus stop traing)
        """

        # `indices` Used in torchdata.Subset
        pool = self.dataset.pool
        print(f"cuda:{torch.cuda.current_device()} inside step() function with pool size of {len(pool)}.")
            
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
        
            # uncertainty_scores = []
            
            # rank, world_size = get_dist_info()
            # if rank == 0:
            #     prog_bar = mmcv.ProgressBar(len(pool))
            
            # for i in range(0, len(pool), self.pred_unit):
            #     # Get logits from segmentation map; 
            #     # dim: (batch_size, 19 classes, img_H, img_W);
            #     end = i + self.pred_unit if i + self.pred_unit < len(pool) else len(pool)
            #     subset = torchdata.Subset(pool, range(i, end))
            #     partial_prob = self.get_probabilities(subset, **self.kwargs)
            #     if partial_prob is not None:
            #         scores = self.heuristic.get_uncertainties(partial_prob).cpu()
            #         uncertainty_scores.extend(scores)
            #     # rank 0 worker will collect progress from all workers.
            #     if rank == 0:
            #         for _ in range(len(subset)):
            #             prog_bar.update()

            rank, world_size = get_dist_info()
            uncertainty_scores = self.get_probabilities(pool, self.heuristic, **self.kwargs)
            # FIXME: Check the line below (has_scores assignment)
            has_scores = uncertainty_scores is not None and uncertainty_scores != []
            if rank == 0 and has_scores:
                ranked = self.heuristic.reorder_indices(uncertainty_scores)
                print(f"ranked size: {ranked.shape}")
                if indices is not None:
                    # use the values in `ranked` to reorder `indices`
                    ranked = indices[np.array(ranked)]
                if len(ranked) > 0:
                    print(f"cuda:{torch.cuda.current_device()} is labeling.")
                    self.dataset.label(ranked[:self.query_size])
                    return True

        return False