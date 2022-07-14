import numpy as np
import warnings
import torch
from torch import Tensor
from collections.abc import Sequence

from .utils import to_prob

available_reductions = {
    "max": lambda x: np.max(x, axis=tuple(range(1, x.ndim))),
    "min": lambda x: np.min(x, axis=tuple(range(1, x.ndim))),
    "mean": lambda x: np.mean(x, axis=tuple(range(1, x.ndim))),
    "sum": lambda x: np.sum(x, axis=tuple(range(1, x.ndim))),
    "none": lambda x: x,
}

def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data


class AbstractHeuristic:
    
    """
    Abstract class that defines a Heuristic with template methods 
    such as get_ranks, compute_score

    Args:
        shuffle_prop (float): shuffle proportion.
        reverse (bool): True if the most uncertain sample has the highest value, and vice versa.
        reduction (Union[str, Callable]): Reduction used after computing the score.
    """

    def __init__(self, shuffle_prop=0.0, reverse=False, reduction="none"):
        self.shuffle_prop = shuffle_prop
        self.reversed = reverse
        assert reduction in available_reductions or callable(reduction)
        self._reduction_name = reduction
        self.reduction = reduction if callable(reduction) else available_reductions[reduction]

    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.
        To be implemented by Child class.
        """
        raise NotImplementedError

    def get_uncertainties_generator(self, predictions):
        """
        Compute the score according to the heuristic.
        Can be used in CombineHeuristics.
        """
        raise NotImplementedError

    def get_uncertainties(self, predictions):

        """ 
        Get the uncertainties.
        Args:   Array of predictions (Tensor)
        Return: Array of uncertainties
        """

        scores = self.compute_score(predictions)
        scores = self.reduction(scores)
        
        if not np.all(np.isfinite(scores)):
            fixed = 0.0 if self.reversed else 10000
            warnings.warn("Some value in scores vector is infinite.")
            scores[~np.isfinite(scores)] = fixed

        return scores

    def reorder_indices(self, scores):
        """
        Re-order indices given their uncertainty score.

        Args:
            scores (ndarray/ List[ndarray]): Array of uncertainties or list of arrays.

        Returns:
            ordered index according to the uncertainty (highest to lowes).

        Raises:
            ValueError if `scores` is not uni-dimensional.
        """
        if isinstance(scores, Sequence):
            scores = np.concatenate(scores)

        if scores.ndim > 1:
            raise ValueError(
                (
                    f"Can't order sequence with more than 1 dimension."
                    f"Currently {scores.ndim} dimensions."
                )
            )
        assert scores.ndim == 1  # We want the uncertainty value per sample.
        ranks = np.argsort(scores) # Ascending order
        if self.reversed:
            ranks = ranks[::-1] # Descending order
        ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks

    def get_ranks(self, predictions):
        """
        Rank the predictions according to their uncertainties.
        Returns:
            Ranked index according to the uncertainty (highest to lowes).
            Scores for all predictions.
        """
        
        scores = self.get_uncertainties(predictions)

        return self.reorder_indices(scores), scores

    def __call__(self, predictions):
        """Rank according to their uncertainties.
        Only return the scores and not the associated uncertainties.
        """
        return self.get_ranks(predictions)[0]

class Random(AbstractHeuristic):

    def __init__(
        self, shuffle_prop=1.0, mode="image",reduction="none", seed=None):
        super().__init__(shuffle_prop=shuffle_prop, reverse=False)

        self.mode = mode
        # rng = random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def compute_score(self, predictions):
        
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        
        # return self.rng.rand(predictions.shape[0])
        
        if self.mode == 'image':
            return self.rng.rand(predictions.shape[0])
        elif self.mode == 'pixel':
            l, c, h, w = predictions.shape
            return self.rng.rand(l, h, w).astype(dtype=np.float16)

class Entropy(AbstractHeuristic):
    """
    Sort by entropy. The higher, the more uncertain.
    """
    def __init__(self, mode, shuffle_prop=0, reduction="none"):
        # reverse = True to turn "ascending" result into a descending order.
        super().__init__(shuffle_prop, reverse=True, reduction=reduction)
        self.mode = mode

    def pixel_mean_entropy(self, softmax_pred):
        bs, classes, img_H, img_W = softmax_pred.size()
        entropy_map = -torch.mul(softmax_pred, softmax_pred.log()).sum(dim=1).half()
        entropy_map = entropy_map.reshape(shape=(bs, img_H, img_W))
        return entropy_map.cpu().numpy()

    def image_mean_entropy(self, softmax_pred):
        bs, classes, img_H, img_W = softmax_pred.size()
        flat_probs = softmax_pred.reshape(bs, -1)
        N = img_H * img_W # number of pixels

        entropy_lst = (-torch.mul(flat_probs, flat_probs.log())).sum(dim=-1) 
        return entropy_lst.cpu().numpy()

    def compute_score(self, predictions):
        """ `predictions` have to be a Tensor """
        probs = to_prob(predictions)
        if self.mode == 'image':
            return self.image_mean_entropy(probs)
        elif self.mode == 'pixel':
            return self.pixel_mean_entropy(probs)

class MarginSampling(AbstractHeuristic):
    """
    Sort by argmin(argmax(prob) - argmax2(prob)). The smaller the
    difference is, the more uncertain. Not that instead of maximizing,
    we want to minimize this value, as opposed to Entropy and Random
    """

    def __init__(self, mode, shuffle_prop=0, reduction="none"):
        super().__init__(shuffle_prop, reverse=False, reduction=reduction)
        self.mode = mode

    def compute_score(self, predictions):
        # FIXME: In development
        probs = to_prob(predictions)
        raise NotImplementedError

        