import numpy as np
import torch
from argparse import Namespace
from mmcv.cnn import build_norm_layer
from mmcv.runner import ModuleList
from mmseg.models.utils import PatchEmbed
from segal.method.mit_modules import TransformerEncoderLayer

def build_embedding(
    cfg: Namespace, mode: str = 'overlap', 
    norm_cfg=dict(type='LN', eps=1e-6)):
    """
    Helper method to build the embedding layers.
    """

    embed_dims = cfg.embed_dims * cfg.num_heads
    if mode == 'non_overlap': # mae branch
        assert not hasattr(cfg, 'stride'), \
            "`stride` argument is not allowed in non_overlap embedding mode."
        return PatchEmbed(
            in_channels=cfg.in_channels,
            embed_dims=embed_dims,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            padding=0, 
            norm_cfg=norm_cfg
        )
    elif mode == 'overlap': # segmentation branch
        return PatchEmbed(
            in_channels=cfg.in_channels,
            embed_dims=embed_dims,
            kernel_size=cfg.patch_size,
            stride=cfg.stride,
            padding=cfg.patch_size // 2, 
            norm_cfg=norm_cfg
        )
    else:
        raise NotImplementedError("unknown `mode`")

def build_projection(
    cfg: Namespace, branch: str, norm_cfg: dict, act_cfg: dict,
    mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate,
    with_cp=False):
    assert branch in ['mae', 'seg']
    total_embed_dims = cfg.embed_dims * cfg.num_heads

    # stochastic num_layer decay rule
    dpr = [ 
        x.item() 
        for x in torch.linspace(0, drop_path_rate, cfg.num_layers) 
    ]
    
    # embedding layer
    if branch == 'mae':
        patch_embed = build_embedding(
            cfg=cfg, mode='non_overlap', norm_cfg=norm_cfg)
    elif branch == 'seg':
        patch_embed = build_embedding(
            cfg=cfg, mode='overlap', norm_cfg=norm_cfg)
    else:
        raise NotImplementedError('unknown `branch`')


    # transformer layer (EfficientMultiheadAttention -> MixFFN)
    encode = ModuleList([
         TransformerEncoderLayer(
            embed_dims=total_embed_dims,
            num_heads=cfg.num_heads,
            feedforward_channels=mlp_ratio * total_embed_dims,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr[idx],
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            sr_ratio=cfg.sr_ratios
         )
         for idx in range(cfg.num_layers)]
    )
    # return_value[0] of build_norm_layer is the name of norm
    norm = build_norm_layer(norm_cfg, total_embed_dims)[1]
    projection_block = ModuleList([patch_embed, encode, norm])
    return projection_block

def get_crop_bbox(img, crop_size):
    """Randomly get a crop bounding box."""
    channels, height, width = img.shape
    margin_h = max(height - crop_size[0], 0)
    margin_w = max(width - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

def get_random_crops(image, crop_size, num=4):

    def crop(img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[..., crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    results = []
    for i in range(num):
        crop_bbox = get_crop_bbox(image, crop_size=crop_size)
        results.append(crop(image, crop_bbox))
    return results
