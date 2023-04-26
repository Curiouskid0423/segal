"""
Module init file
"""

from .active_loop import ActiveLearningLoop
from .dataset import ActiveLearningDataset
from . import heuristics

def get_heuristics(mode, name, **kwargs):

    """
    Args:
        mode:           sample mode (e.g. image, pixel)
        name:           Name of the heuristic.
        kwargs:         Complementary arguments.

    Returns:
        AbstractHeuristics instance
    """

    heuristic_dic = {
        "random": heuristics.Random,
        "entropy": heuristics.Entropy,
        "margin": heuristics.MarginSampling,
        "ripu": heuristics.RegionImpurity,
    }

    assert mode != None, "mode argument has to be either pixel or image for sampling"
    return heuristic_dic[name](mode = mode, **kwargs)

