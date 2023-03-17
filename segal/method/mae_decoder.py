import os
import os.path as osp
import torch
import torch.nn as nn
import einops
from mmcv.runner import force_fp32, get_dist_info, ModuleList
from mmcv.cnn import ConvModule
from mmseg.utils import get_root_logger
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from segal.method import utils
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

        self.save_results = True
        self.curr_mae_iter = 0
        self.save_mae_iter = (2975 // 8) * 2

        # MiT-based fusion decoder (to be completed)
        self.use_vit = False
        if self.use_vit:
            nh = 4 # num heads
            self.crop_size = 160 # crop size
            self.patch_size = 8
            decoder_mlp_ratio = 4
            attn_embed_dims = self.channels
            self.transformer = TransformerEncoderLayer(
                embed_dims=attn_embed_dims,
                num_heads=nh,
                feedforward_channels=attn_embed_dims*decoder_mlp_ratio,
                drop_path_rate=0.1,
                sr_ratio=1
            ) # (batch, num_patch**2, embed_dims)
            self.channel_compression = ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg)
            self.decode_projection = nn.Linear(
                attn_embed_dims,
                self.patch_size*self.patch_size*self.num_classes, 
                bias=True) 
        else:
            # CNN-based fusion decoder (16, 256*4, 120, 120)
            self.fusion_decoder = ConvModule(
                in_channels=self.channels,
                out_channels=self.num_classes,
                kernel_size=1,
                norm_cfg=self.norm_cfg
            )
        
        
    @force_fp32(apply_to=('rec_logit', 'ori_image'))
    def losses(self, rec_logit, ori_image, mask):
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
            mask.unsqueeze(dim=1).expand(-1, 3, -1, -1), 
            size=rec_logit.shape[2:], 
            mode='nearest'
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
                )
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = numeric_loss
            else:
                loss[loss_decode.loss_name] += numeric_loss

        return loss


    def forward_cnn(self, inputs, mask_tokens, ids_restore, original_size):

        outs = []

        # Append mask tokens, if needed, before feeding into the decoder
        # This method currently has dimension error
        if mask_tokens != None:
            restored_patches = []
            for entry in inputs:
                batch, channels, _, feature_size = entry.shape
                curr_mask = mask_tokens.expand(batch, -1, -1)
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
                    input=conv(x), size=original_size,
                    mode=self.interpolate_mode, align_corners=self.align_corners
                ))

        outs = torch.cat(outs, dim=1) 
        out = self.fusion_decoder(outs) 
        return out

    def forward_vit(self, inputs, mask_tokens, ids_restore):

        # Concat and reshape the 4 layer features
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx] # upsample 1x1 convolutions
            outs.append(conv(x))
           
        outs = torch.cat(outs, dim=1)
        outs = self.channel_compression(outs) 
        assert mask_tokens != None

        # reshape and preprocessing
        mask_tokens = mask_tokens.expand(len(outs), -1, -1)
        outs = einops.rearrange(outs, 'b c n f -> b n (c f)')
        feats = restore_masked(kept_x=outs, masked_x=mask_tokens, ids_restore=ids_restore)
        _, total_patches, _ = feats.shape
        num_patch = int(total_patches**0.5)
        hw_shape = (num_patch, num_patch)
        # forward pass
        feats = self.transformer(feats, hw_shape, sr_enable=False)  # (batch, n_patch**2, feat_size)
        feats = self.decode_projection(feats) 
        out = einops.rearrange(
            feats, 'b (nw nh) (c pw ph) -> b c (nw pw) (nh ph)', 
            c=self.num_classes, nh=num_patch, nw=num_patch, pw=self.patch_size, ph=self.patch_size)

        return out
        

    def forward(self, encodings, mask_tokens=None, ids_restore=None, ori_shape=None):
        """
        To be edited.
        """
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(encodings)
        original_size = ori_shape if ori_shape != None else inputs[0].shape[2:]
        
        if not self.use_vit:
            return self.forward_cnn(inputs, mask_tokens, ids_restore, original_size)
        else:
            return self.forward_vit(inputs, mask_tokens, ids_restore)


    # override the BaseDecodeHead
    def forward_train(
        self, inputs, mask, img_metas, ids_restore, ori_images, train_cfg):
        """
        Forward function for training for mask reconstruction

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        append_masks = True if len(inputs) == 2 else False
        logger = get_root_logger()
        if append_masks:
            encoding, mask_tokens = inputs
            seg_logits = self(
                encoding, mask_tokens, ids_restore, ori_shape=ori_images.shape[-2:])
        else:
            seg_logits = self(inputs, ori_shape=ori_images.shape[-2:])

        # visualize reconstructed images
        rank, _ = get_dist_info()
        if self.save_results and rank == 0 and self.curr_mae_iter % self.save_mae_iter == 0:
            logger.info(f"saving reconstructed images at MAE iteration {self.curr_mae_iter}...")
            cwd = os.getcwd()
            save_path = osp.join(cwd, 'reconstructed_images', f'ep{self.curr_mae_iter}')
            utils.save_reconstructed_images(
                path=save_path, 
                ori=ori_images.detach(), 
                rec=seg_logits.detach(),
                img_metas=img_metas)
        self.curr_mae_iter += 1

        losses = self.losses(seg_logits, ori_images, mask)
        return losses


