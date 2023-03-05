"""
Full model file
"""
from argparse import Namespace
import math
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import nlc_to_nchw
from segal.method.mit import MixVisionTransformer
from segal.method.utils import build_projection

@BACKBONES.register_module()
class TwinMixVisionTransformer(BaseModule):

    """
    A MixVisionTransformer BACKBONE class with separate projection heads.
    Note that SEGMENTOR class should take care of mask autoencoding. This 
    file only takes care of forward pass logic regardless of batch preprocessing
    """

    def __init__(self, 
        seg_projection: Namespace,
        mae_projection: Namespace,
        encoder: Namespace,
        mlp_ratio,
        qkv_bias,
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', eps=1e-6),
        pretrained=None,
        init_cfg=None,
        with_cp=False):
                
        super(TwinMixVisionTransformer, self).__init__(init_cfg=init_cfg)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'

        # hyperparameters for the model
        self.num_subimages = 256 # FIXME: should be set in config file
        self.subimages_crop_size = (384, 384) # FIXME: should be set in config file

        # save configs
        self.cfg_seg = seg_projection
        self.cfg_mae = mae_projection

        # Transformer Block 1 is separate for the two branches
        self.seg_projection = build_projection(
            self.cfg_seg, 'seg', norm_cfg, act_cfg, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, with_cp)
        self.mae_projection = build_projection(
            self.cfg_mae, 'mae', norm_cfg, act_cfg, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate, with_cp)
        
        # Main shared weight backbone
        assert self.cfg_seg.embed_dims * self.cfg_seg.num_heads \
            == self.cfg_mae.embed_dims * self.cfg_mae.num_heads
        enc_in_channel = self.cfg_seg.embed_dims * self.cfg_seg.num_heads
        self.encoder = MixVisionTransformer(
            in_channels=enc_in_channel,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            with_cp=with_cp,
            **encoder
        )


    def init_weights(self):
        """ Copied from MixVisionTransformer class """
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(TwinMixVisionTransformer, self).init_weights()

    def projection(self, x, ptype='seg'):
        if ptype == 'seg':
            proj_head = self.seg_projection
        elif ptype == 'mae':
            proj_head = self.mae_projection
        else:
            raise NotImplementedError("unknown ptype")

        patchify, encode, normalize = proj_head
        x, hw_shape = patchify(x)
        for block in encode:
            x = block(x, hw_shape)
        x = normalize(x)
        # convert embeddings into sparital tensor
        x = nlc_to_nchw(x, hw_shape)
        return x

    def forward(self, x, train_type='seg'):

        assert train_type in ['mae', 'seg']

        feat = self.projection(x, ptype=train_type)
        pred = self.encoder(feat)
        all_features = [feat] + pred
        return all_features