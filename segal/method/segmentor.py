# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors import EncoderDecoder


@SEGMENTORS.register_module()
class MultitaskSegmentor(EncoderDecoder):
    """
    A special segmentor structure such that we can train in alternation.
    Currently expects backbone to be `TwinMixVisionTransformer` and 
    decode

    
    """

    def __init__(self,
        backbone,
        decode_head,
        auxiliary_head,
        rec_crop_size,
        heuristics='entropy',
        neck=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None):

        assert neck is None, \
            "`neck` argument is not reliably supported by MultitaskSegmentor currently."

        # backbone:        TwinMixVisionTransformer
        # decode_head:     SegformerHead (segmentation)
        # auxiliary_head:  SegformerHead (mae)
        self.rec_crop_size = rec_crop_size
        self.heuristics = heuristics
        super(MultitaskSegmentor, self).__init__(
            backbone, decode_head, auxiliary_head, 
            train_cfg, test_cfg, pretrained, init_cfg=init_cfg)
        
            
    # def sample_subimages(self, x, stage: str = 'warmup', num_sample: int = 2):
    #     """ 
    #     Given an image x, sample `num_sample` sub-images. If sample stage 
    #     is 'warmup', simply sample randomly. If the sample stage is 'train', 
    #     sample by self.heuristics.
    #     """
    #     assert stage in ['warmup', 'train']
    #     assert hasattr(self, "heuristics")

    def extract_feat(self, img, train_type: str):
        """ 
        Extract features from images using the 
        right backbone according to train_type. 
        """
        x = self.backbone(img, train_type)
        return x

    # def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for decode head in
    #     training."""
    #     losses = dict()
    #     loss_decode = self.decode_head.forward_train(x, img_metas,
    #                                                  gt_semantic_seg,
    #                                                  self.train_cfg)

    #     losses.update(add_prefix(loss_decode, 'decode'))
    #     return losses

    def forward_train(self, img, img_metas, gt_semantic_seg, stage='seg'):
        """Overrides EncoderDecoder. Forward function for training.

        when `train_type` is 'mae', return reconstructed image
        when `train_type` is 'seg' return the predicted segmentation map
        at inference, the train_type will be 'seg'
        
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert stage in ['mae', 'seg']
        
        x = self.extract_feat(img, train_type=stage)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        return losses


    def train_step(self, data_batch, optimizer, **kwargs):
        """Override BaseSegmentor.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch) # calls `forward_train`
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

        
    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output