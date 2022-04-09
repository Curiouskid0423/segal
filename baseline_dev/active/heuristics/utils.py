"""
Utility files
"""
import numpy as np
from scipy.special import softmax, expit

def to_prob(probs: np.ndarray):
    """
    If the probabilities array is not a distrubution will softmax it.

    """
    not_bounded = np.min(probs) < 0 or np.max(probs) > 1.0
    multiclass = probs.shape[1] > 1
    sum_to_one = np.allclose(probs.sum(1), 1)
    if not_bounded or (multiclass and not sum_to_one):
        probs = softmax(probs, 1) if multiclass else expit(probs)
            
    return probs