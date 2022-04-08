"""
Utility files
"""
import numpy as np
from scipy.special import softmax, expit

def to_prob(probabilities: np.ndarray):
    """
    If the probabilities array is not a distrubution will softmax it.

    Args: probabilities (array): [batch_size, num_classes, ...]
    Returns: "softmaxed" probabilities.
    """
    not_bounded = np.min(probabilities) < 0 or np.max(probabilities) > 1.0
    multiclass = probabilities.shape[1] > 1
    sum_to_one = np.allclose(probabilities.sum(1), 1)
    if not_bounded or (multiclass and not sum_to_one):
        if multiclass:
            probabilities = softmax(probabilities, 1)
        else:
            probabilities = expit(probabilities)
    return probabilities