# Originally from https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
# From mmseg segformer_head.py file


import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads import SegformerHead
from mmseg.ops import resize

@HEADS.register_module()
class MAESegformerHead(SegformerHead):
    """
    An extended SegformerHead to integrate MAE objective.
    forward() will return reconstructed images.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super(MAESegformerHead, self).__init__(
            interpolate_mode=interpolate_mode, **kwargs)
    
    @force_fp32(apply_to=('rec_logit'))
    def losses(self, rec_logit, ori_image):
        """ Compute reconstruction loss """
        loss = dict() # will be logged

        # dimension adjustment
        rec_logit = resize(
            input=rec_logit,
            size=ori_image.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(rec_logit, ori_image)
        else:
            seg_weight = None

        # compute loss
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            numeric_loss = loss_decode(
                    rec_logit,
                    ori_image,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = numeric_loss
            else:
                loss[loss_decode.loss_name] += numeric_loss

        return loss


