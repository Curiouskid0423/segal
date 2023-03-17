# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from argparse import Namespace
import einops

from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS
from mmseg.utils import get_root_logger
from mmseg.models.segmentors import EncoderDecoder
from segal.method.utils import get_random_crops
from segal.utils.masking import (patchify, random_masking, get_shuffled_ids, 
                                restore_masked, unpatchify)


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
        backbone,
        decode_head,
        auxiliary_head,
        heuristics='entropy',
        neck=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None):

        assert neck is None, \
            "`neck` argument is not reliably supported by MultiTaskSegmentor currently."


        self.logger = get_root_logger()
        # save config variables
        self.mae_configs = Namespace(**backbone['mae_projection'])
        self.mae_decoder_configs = Namespace(**auxiliary_head)
        self.seg_configs = Namespace(**backbone['seg_projection'])

        # mae required variables 
        self.rec_crop_size = self.mae_configs.rec_crop_size
        self.heuristics = heuristics

        super(MultiTaskSegmentor, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=None,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        # set num_patches in backbone
        self.mask_ratio = self.mae_configs.mask_ratio
        decoder_embedding_dim = self.mae_decoder_configs.channels 
        # decoder_embedding_dim = 1 # FIXME: not expressive enough
        self.mask_tokens = nn.Parameter(torch.randn(1, 1, decoder_embedding_dim) * 0.02)
        
        mae_patch = self.mae_configs.patch_size
        assert self.rec_crop_size[0] % mae_patch == 0 

        self.num_patches = self.rec_crop_size[0] // mae_patch
        self.masked_length = int(self.num_patches * self.num_patches * self.mask_ratio)
        self.keep_length = self.num_patches*self.num_patches - self.masked_length

    def sample_subimages(self, x, stage: str = 'warmup', num_sample: int = 2):
        """ 
        Given an image x, sample `num_sample` sub-images. If sample stage 
        is 'warmup', simply sample randomly. If the sample stage is 'train', 
        sample by self.heuristics.
        """
        assert stage in ['warmup', 'train']
        if stage == 'warmup':
            # sample by random
            result = []
            for image in x:
                # image -> (3, 264, 528)
                list_of_crops = get_random_crops(
                    image=image, crop_size=self.rec_crop_size, num=num_sample)
                result.extend(list_of_crops) # [2*4, (256, 256)]
        else:
            # sample by heuristics
            raise NotImplementedError('mask-by-heuristics is not implemented yet')
        
        return torch.stack(result, dim=0) 
        
    def extract_feat(self, img, train_type:str='seg'):
        """ 
        Extract features from images using `TwinMixVisionTransformer`,
        which takes care of different types of projections based on train_type 
        """
        x = self.backbone(img, train_type)
        return x

    def forward_train(self, img, img_metas, gt_semantic_seg, stage='seg'):
        """Overrides EncoderDecoder. Forward function for training.

        when `train_type` is 'mae', return reconstructed image
        when `train_type` is 'seg' return the predicted segmentation map
        
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
        
        losses = dict()
        truncate_mask_at_encode = True # a boolean value to decide whether the encoder takes in masks

        if stage == 'mae':
            subimages = self.sample_subimages(img, num_sample=4) # 16 # (B, C, H, W)
            batch_size = len(subimages)
            ids_shuffle = get_shuffled_ids(
                batch_size=batch_size, 
                total_patches=self.num_patches*self.num_patches
            ) # (B, H*W)
            
            # patchify: (8, 3, 256, 256) -> (8, 3, num_patches, patch_size*patch_size)
            mae_patch_size = self.mae_configs.patch_size
            batch_of_patches = patchify(subimages, mae_patch_size)
            # fix to append channels into the feature length
            visible_patches, mask, ids_restore = \
                random_masking(batch_of_patches, self.keep_length, ids_shuffle) # (b c l f)
            if truncate_mask_at_encode:
                # assumes that the model does not have any overlapping patchify module
                masked_images = einops.rearrange(
                    visible_patches, 'b c l (ph pw) -> b c (l ph) pw', ph=mae_patch_size, pw=mae_patch_size)
            else:
                b, c, _, f = visible_patches.shape
                assert self.mask_tokens.shape[-1] == 1, 'fix mask token length problem before proceeding'
                visible_patches = einops.rearrange(visible_patches, 'b c l f -> b l (c f)')
                masked_images = restore_masked(
                    visible_patches, 
                    masked_x=self.mask_tokens.expand(b, self.masked_length, c*f),
                    ids_restore=ids_restore
                )

                masked_images = einops.rearrange(
                    masked_images, 'b l (c f) -> b c l f', 
                    c=self.mae_configs.in_channels, f=mae_patch_size**2)
                masked_images = unpatchify(masked_images, patch_size=mae_patch_size)

            mae_encoding = self.extract_feat(masked_images, train_type='mae') 

            # append the mask tokens back, predict pixel-level reconstruction
            mask = einops.rearrange(mask, 'b (h w) -> b h w', h=self.num_patches, w=self.num_patches)

            if truncate_mask_at_encode:
                loss_decode = self._mae_decoder_forward_train(
                    encoding=mae_encoding, 
                    mask_tokens=self.mask_tokens.expand(-1, self.masked_length, -1),
                    mask=mask, 
                    img_metas=img_metas,
                    ids_restore=ids_restore, 
                    ori_images=subimages.detach()
                )
            else:
                loss_decode = self._mae_decoder_forward_train(
                    encoding=mae_encoding, mask=mask, img_metas=img_metas,
                    ids_restore=ids_restore, ori_images=subimages.detach()
                )

        else:
            feat = self.extract_feat(img, train_type='seg')
            loss_decode = self._decode_head_forward_train(
                feat, img_metas, gt_semantic_seg)

        losses.update(loss_decode)
        
        return losses

    def _mae_decoder_forward_train(
        self, encoding, mask, img_metas, ids_restore, ori_images, mask_tokens=None):

        losses = dict()
        loss_decode = self.auxiliary_head.forward_train(
            inputs=(encoding) if mask_tokens is None else (encoding, mask_tokens),
            mask=mask,
            img_metas=img_metas,
            ids_restore=ids_restore,
            ori_images=ori_images,
            train_cfg=self.train_cfg
        )
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