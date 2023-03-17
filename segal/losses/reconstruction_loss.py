import torch
import torch.nn as nn

from mmseg.models.builder import LOSSES
from mmseg.models.losses import weighted_loss

@weighted_loss
def rec_loss(pred, target, mask):
    assert pred.size() == target.size() and target.numel() > 0
    loss = (pred - target) ** 2 * mask
    return loss

@LOSSES.register_module
class ReconstructionLoss(nn.Module):

    def __init__(self, loss_name='loss_rec',reduction='mean', loss_weight=1.0, avg_non_ignore=False):
        super(ReconstructionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self,
                pred,
                target,
                mask,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = rec_loss(pred, target, reduction=reduction, mask=mask, avg_factor=mask.sum())
        loss = self.loss_weight * loss
        return loss