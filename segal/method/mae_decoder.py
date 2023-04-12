import os
import os.path as osp
import torch
import torch.nn as nn
import einops
from mmcv.runner import force_fp32, ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_, trunc_normal_init)

from mmseg.utils import get_root_logger
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.ops import resize

from segal.utils.masking import restore_masked, patchify


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

    def __init__(self, 
                 in_channels,
                 num_layers, 
                 num_heads,
                 img_size,
                 patch_size,
                 embed_dims=768,
                 mlp_ratio=4,
                 drop_path_rate=0.1,
                 attn_drop_rate=0.0,
                 drop_rate=0.0,
                 qkv_bias=True,
                 num_fcs=2,
                 with_cp=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 interpolate_mode='bicubic',
                 init_std=0.02,
                 **kwargs):
        super(MaskDecodeHead, self).__init__(
            in_channels=in_channels, **kwargs)
        
        assert self.num_classes == 3, \
            "Assuming the input is RGB, num_classes has to be 3 in MaskDecodeHead."
            
        self.logger = get_root_logger()
        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.num_layers = num_layers
        self.save_results = True

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))
        
        self.projection = nn.Linear(in_channels, embed_dims)
        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.upsampler = nn.Linear(
                embed_dims, patch_size*patch_size*self.num_classes, bias=True) 
        
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.num_patches = \
            (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.init_std = init_std
        delattr(self, 'conv_seg')
        
    def init_weights(self):
        # FIXME: double check if the init_weights is implemented correctly for pretraining
        module_name = self.__class__.__name__
        self.logger.info(f'initialize {module_name} with init_cfg {self.init_cfg}')
            
        trunc_normal_init(self.projection, std=self.init_std)
        trunc_normal_init(self.upsampler, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def _pos_embedding(self, patched_img, hw_shape, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == self.num_patches + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    @force_fp32(apply_to=('rec_logit', 'ori_image'))
    def losses(self, rec_logit, ori_image, mask):
        """ Compute reconstruction loss """
        loss = dict() # will be logged

        # compute loss
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            numeric_loss = loss_decode(
                    rec_logit, ori_image, mask=mask)
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = numeric_loss
            else:
                loss[loss_decode.loss_name] += numeric_loss

        return loss


    def forward_vit(self, inputs, mae_args, test_mode=False):

        # forward pass
        feats = self.projection(inputs)

        # reshape and preprocessing
        B, L, F = inputs.shape
        ids_restore = mae_args['ids_restore']
        mask_tokens = self.mask_tokens.repeat(B, ids_restore.shape[1] - L, 1)
        feats = restore_masked(kept_x=feats, masked_x=mask_tokens, ids_restore=ids_restore)
        _, total_patches, _ = feats.shape
        num_patch = int(total_patches**0.5)
        
        feats = self._pos_embedding(feats, (num_patch, num_patch), self.pos_embed)

        for layer in self.layers:
            feats = layer(feats)
        feats = self.decoder_norm(feats)
        feats = self.upsampler(feats)
        
        if test_mode:
            out = einops.rearrange(
                feats, 'b (nw nh) (pw ph c) -> b c (nw pw) (nh ph)', 
                c=self.num_classes, nh=num_patch, nw=num_patch, pw=self.patch_size, ph=self.patch_size)
            return out
        else:
            return feats
        

    def forward(self, encodings, mae_args=None, test_mode=False):
        """
        To be edited.
        """
        inputs = self._transform_inputs(encodings)
        return self.forward_vit(inputs, mae_args, test_mode)


    # override the BaseDecodeHead
    def forward_train(
        self, inputs, mae_args, train_cfg):
        """
        Forward function for training for mask reconstruction

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        rec_logits = self(inputs, mae_args)
        ground_truth = patchify(mae_args['ori_images'], self.patch_size)
        losses = self.losses(rec_logits, ground_truth, mae_args['masks'])
        return losses
