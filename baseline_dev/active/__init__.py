"""
Module init file
"""

from .active_loop import ActiveLearningLoop
from .dataset import ActiveLearningDataset
from . import heuristics

def get_heuristics(name, shuffle_prop=0., reduction="none", **kwargs):

    """
    Args:
        name:           Name of the heuristic.
        shuffle_prop:   Shuffling proportion when getting ranks.
        reduction:      Reduction used after computing the score.
        kwargs:         Complementary arguments.

    Returns:
        AbstractHeuristics instance
    """

    heuristic_dic = {
        "random": heuristics.Random,
        "entropy": heuristics.Entropy,
    }

    return heuristic_dic[name](shuffle_prop=shuffle_prop, reduction=reduction, **kwargs)

