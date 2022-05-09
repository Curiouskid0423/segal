"""
Utility files
"""
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax, expit

def to_prob(probs: Union[np.ndarray, torch.Tensor]):
    """
    If the probabilities array is not a distrubution will softmax it.
    probs from entropy function: (batch_size, class, img_H, img_W)
    """
    if not isinstance(probs, torch.Tensor):
        return to_prob_numpy(probs)
    bs, classes, img_H, img_W = probs.size()
    flat_probs = probs.view(bs, classes, -1)
    not_bounded = torch.min(probs) < 0 or torch.max(probs) > 1.0
    multiclass = probs.size()[1] > 1
    
    device = f"cuda:{probs.get_device()}" if probs.get_device() >= 0 else 'cpu'
    one_target = torch.Tensor([[1.]]).to(device)
    sum_to_one = torch.allclose(probs.sum(axis=(-1, -2)), one_target)

    if not_bounded or (multiclass and not sum_to_one):
        # Softmax all pixel values
        probs = F.softmax(flat_probs, dim=-1).reshape(bs, classes, img_H, img_W)
        
    return probs

def to_prob_numpy(probs):
    not_bounded = np.min(probs) < 0 or np.max(probs) > 1.0
    multiclass = probs.shape[1] > 1
    sum_to_one = np.allclose(probs.sum(axis=(-1, -2)), 1)
    if not_bounded or (multiclass and not sum_to_one):
        # Softmax all pixel values
        probs = softmax(probs, axis=(-1, -2))
        
    return probs