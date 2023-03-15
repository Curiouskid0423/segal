import torch
import torch.nn as nn
import einops
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, ConvTranspose2d

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from segal.method.mit_modules import TransformerEncoderLayer
from segal.utils.masking import restore_masked


@HEADS.register_module()
class MaskDecodeHead(BaseDecodeHead):
    """
    An extended SegformerHead to integrate MAE objective. MaskDecodeHead
    will take in the embeddings of the visible patches and patch_ids to 
    restore the order of patches. Then, append the mask tokens to the 
    corresponding position, pass through a decoder backbone, and eventually
    return a reconstruction loss in forward_train(). 
    forward() will return reconstructed images.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super(MaskDecodeHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        
        assert self.num_classes == 3, "Assuming the input is RGB, num_classes has to be 3 in MaskDecodeHead."
        
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # CNN-based fusion decoder
        self.fusion_decoder = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.num_classes,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # MiT-based fusion decoder (to be completed)
        # self.fusion_decoder = TransformerEncoderLayer(
        #     embed_dims=32,
        #     num_heads=None,
        #     feedforward_channels=None,
        #     drop_rate=None,
        #     attn_drop_rate=None,
        #     drop_path_rate=None
        # )
    
    @force_fp32(apply_to=('rec_logit', 'ori_image'))
    def losses(self, rec_logit, ori_image, mask_tokens):
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

        # resize masks
        mask_by_pixels = resize(
            mask_tokens.unsqueeze(dim=1).expand(-1, 3, -1, -1), 
            size=rec_logit.shape[2:], 
            mode=self.interpolate_mode, 
            align_corners=self.align_corners
        )

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
                    mask=mask_by_pixels,
                    # ignore_index=self.ignore_index
                )
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = numeric_loss
            else:
                loss[loss_decode.loss_name] += numeric_loss

        return loss

    def forward(self, encodings, mask_tokens=None, ids_restore=None, ori_shape=None):
        """
        To be edited.
        """
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(encodings)
        outs = []
        original_size = ori_shape if ori_shape != None else inputs[0].shape[2:]

        # Append mask tokens, if needed, before feeding into the decoder
        if mask_tokens != None:
            restored_patches = []
            for entry in inputs:
                batch, channels, _, feature_size = entry.shape
                curr_mask = mask_tokens.expand(batch, -1, channels*feature_size)
                entry = einops.rearrange(entry, 'b c l f -> b l (c f)')
                restored = restore_masked(
                    kept_x=entry, masked_x=curr_mask, ids_restore=ids_restore)
                restored = einops.rearrange(
                    restored, 'b l (c f) -> b c l f', c=channels, f=feature_size)
                num_patches = int(restored.shape[2] ** 0.5)
                restored = restored.reshape(batch, -1, num_patches, num_patches)
                restored_patches.append(restored)
            inputs = restored_patches

        # Concat and reshape the 4 layer features
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx] # upsample 1x1 convolutions
            outs.append(
                resize(
                    input=conv(x),
                    size=original_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners)
                )

        outs = torch.cat(outs, dim=1) # include_at_encode -> torch.Size([16, 1024, 15, 15])
        out = self.fusion_decoder(outs) 
        # "out" for append_at_decode = (16, 1, 225, 1)
        # "out" for include_at_encode = (16, 1, 15, 15) FIXED
        # ori_images = (16, 3, 120, 120)
        return out


    # override the BaseDecodeHead
    def forward_train(
        self, inputs, mask, img_metas, ids_restore, ori_images, train_cfg):
        """
        Forward function for training for mask reconstruction

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        append_masks = True if len(inputs) == 2 else False
        if append_masks:
            encoding, mask_tokens = inputs
            seg_logits = self(
                encoding, mask_tokens, ids_restore, ori_shape=ori_images.shape[-2:])
        else:
            seg_logits = self(inputs, ori_shape=ori_images.shape[-2:])
        losses = self.losses(seg_logits, ori_images, mask)
        return losses


