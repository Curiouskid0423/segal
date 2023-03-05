import torch
import torch.nn as nn

from mmseg.models.builder import LOSSES
from mmseg.models.losses import weighted_loss

@weighted_loss
def rec_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = (pred - target) ** 2
    return loss

@LOSSES.register_module
class ReconstructionLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ReconstructionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * rec_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss