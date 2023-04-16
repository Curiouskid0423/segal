# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS
from mmseg.utils import get_root_logger
from mmseg.models.segmentors import EncoderDecoder


@SEGMENTORS.register_module()
class MultiTaskSegmentor(EncoderDecoder):

    """
    A special segmentor structure such that we can train in alternation.
    Currently expects backbone to be `TwinMixVisionTransformer` with 
    decode_head `SegformerHead` and auxiliary_head `MaskDecodeHead`. 
    SEGMENTORS class takes care of 
    (1) `forward()` logic in ALL 
        - train_step()
        - val_step() 
        - inference()
    (2) loss computation

    The training in alternation logic should be taken care of in the
    MultitaskActiveRunner class. 

    """

    def __init__(self,
                 mae_config,
                 backbone,
                 decode_head,
                 auxiliary_head,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        assert neck is None, \
            "`neck` argument is not reliably supported by MultiTaskSegmentor currently."

        self.logger = get_root_logger()
        self.mae_configs = mae_config
        self.mask_ratio = self.mae_configs.mask_ratio
        H, W = backbone.img_size
        P = backbone.patch_size
        self.num_patches = (W // P) *  (H // P)
                
        super(MultiTaskSegmentor, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=None,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        
    def extract_feat(self, img, mae_args=None):
        """ 
        Extract features from images using `TwinMixVisionTransformer`,
        which takes care of different types of projections based on train_type 
        """
        x = self.backbone(img, mae_args)
        return x

    def forward_train(self, img, img_metas, gt_semantic_seg, stage='multitask'):
        """Overrides EncoderDecoder. Forward function for training.
        
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

        assert stage in ['mae', 'multitask']
        
        losses = dict()

        if stage == 'mae':
            mae_encoding, masks, ids_restore = self.extract_feat(
                img, mae_args={ 'ratio': self.mask_ratio, }) 

            loss_decode = self._mae_decoder_forward_train(
                encoding=mae_encoding, 
                mae_args={
                    'masks': masks,
                    'ids_restore': ids_restore,
                    'img_metas': img_metas,
                    'ori_images': img.detach()
                }
            )
            losses.update(loss_decode)

        else:
            
            # use_mae = False
            use_mae = True

            feat = self.extract_feat(img)        
            loss_decode = self._decode_head_forward_train(
                feat, img_metas, gt_semantic_seg)
            losses.update(loss_decode)

            if use_mae and self.with_auxiliary_head:
                mae_encoding, masks, ids_restore = self.extract_feat(
                    img, mae_args={ 'ratio': self.mask_ratio, })
                loss_aux = self._mae_decoder_forward_train(
                    encoding=mae_encoding, 
                    mae_args={
                        'masks': masks,
                        'ids_restore': ids_restore,
                        'img_metas': img_metas,
                        'ori_images': img.detach()
                    }
                )
                losses.update(loss_aux)

        return losses

    def _mae_decoder_forward_train(self, encoding, mae_args):

        losses = dict()
        loss_decode = self.auxiliary_head.forward_train(
            inputs=encoding, mae_args=mae_args, train_cfg=self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def train_step(self, data_batch, stage, optimizer, **kwargs):
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
        # calls `forward_train`. by default, return_loss=True.
        losses = self(stage=stage, **data_batch) 
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs
    
    def mae_inference(self, img: torch.Tensor, return_mask=False):
        with torch.no_grad():
            if len(img) == 3:
                img = img.unsqueeze(0)
            mae_encoding, masks, ids_restore = self.extract_feat(
                img, mae_args={ 'ratio': self.mask_ratio, }) 
            out = self.auxiliary_head.forward(
                mae_encoding, mae_args={'ids_restore': ids_restore}, test_mode=True)
            if return_mask:
                return out, masks
            else:
                return out