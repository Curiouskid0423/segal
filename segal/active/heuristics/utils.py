"""
Utility files
"""
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import math

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
    sum_to_one = torch.allclose(probs.sum(axis=(-1, -2)), one_target.to(probs.dtype))

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

class RIPU_Net(nn.Module):

    def __init__(self, size, channels, padding='zeros'):
        super(RIPU_Net, self).__init__()
        assert size % 2 == 1, "size of the receptive field has to be an odd number."
        self.categories = channels
        
        # Modules for just region-impurity
        self.purity_conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=size,
            stride=1, 
            padding= size//2,
            bias=False,
            padding_mode=padding, 
            groups=channels # each channel should have a separate filter
        ) # (batch, 19, 256, 512) -> (256, 512)
        weight = torch.ones((1, 1, size, size), dtype=torch.float32) # (C_in, C_out, K, S)
        weight = weight.repeat([channels, 1, 1, 1])
        self.purity_conv.weight = nn.Parameter(weight)
        self.purity_conv.requires_grad_(False)

        # Modules for region-entropy
        self.entropy_conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=size,stride=1, 
            padding=size//2, bias=False, padding_mode=padding
        )
        weight = torch.ones((1, 1, size, size), dtype=torch.float32)
        self.entropy_conv.weight = nn.Parameter(weight)
        self.entropy_conv.requires_grad_(False)

    def forward(self, pred, use_entropy):

        """
        Take in an softmax prediction map of shape (B, C, H, W), and output a 
        region-impurity score map with the same size.
        """
        get_entropy = lambda p, dim, keepdim: torch.sum(-p * torch.log(p + 1e-6), dim=dim, keepdim=keepdim)

        maps = pred.argmax(dim=1).squeeze(axis=1) # pseudo-label maps (B, H, W)

        assert len(maps.shape) == 3, "input shape should be 3 dimension (B, H, W) category map."
        scores = []

        for idx, map in enumerate(maps):
            one_hot = F.one_hot(map, num_classes=self.categories).float() # (H, W, C)
            one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0) # (1, C, H, W)
            summary = self.purity_conv(one_hot) # (1, C, H, W) | each value 
            total_categories = torch.sum(summary, dim=1, keepdim=True)  # (1, 1, H, W)
            dist = summary / total_categories  # (1, 19, H, W)
            unnormalized = get_entropy(dist, dim=1, keepdim=True)
            score = unnormalized / math.log(self.categories) # (1, 1, H, W)     

            if use_entropy:
                # multiply entropy onto score
                pred = pred[idx].squeeze(axis=0) # (C, H, W)
                unnormalized = get_entropy(pred, dim=0, keepdim=False)[None, None, ...]
                pixel_entropy = unnormalized / math.log(self.categories)  # (1, 1, H, W)     
                region_sum_entropy = self.entropy_conv(pixel_entropy)  # (1, 1, H, W)
                prediction_uncertainty = region_sum_entropy / total_categories # (1, 1, H, W)
                score *= prediction_uncertainty
            scores.append( score.cpu().squeeze(dim=0).squeeze(dim=0).numpy() )
        return np.array(scores)