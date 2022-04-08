"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from dataset import ActiveLearningDataset
from typing import Callable
import heuristics

class ActiveLearningLoop:

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        **kwargs,
    ):
        pass