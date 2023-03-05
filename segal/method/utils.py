import torch
from argparse import Namespace
from mmcv.cnn import build_norm_layer
from mmcv.runner import ModuleList
from mmseg.models.utils import PatchEmbed, nlc_to_nchw
from segal.method.mit_modules import TransformerEncoderLayer

def index_sequence(x: torch.Tensor, ids):
    """Index tensor (x) with indices given by ids
    Args:
        x: input sequence tensor, can be 2D (batch x length) or 3D (batch x length x feature)
        ids: 2D indices (batch x length) for re-indexing the sequence tensor
    """
    if len(x.shape) == 3:
        ids = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.take_along_dim(x, ids, dim=1)

def build_embedding(
    cfg: Namespace, mode: str = 'overlap', 
    norm_cfg=dict(type='LN', eps=1e-6)):
    """
    Helper method to build the embedding layers.
    """

    embed_dims = cfg.embed_dims * cfg.num_heads
    if mode == 'non_overlap':
        assert not hasattr(cfg, 'stride'), \
            "`stride` argument is not allowed in non_overlap embedding mode."
        return PatchEmbed(
            in_channels=cfg.in_channels,
            embed_dims=embed_dims,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            padding=cfg.patch_size // 2, 
            norm_cfg=norm_cfg
        )
    elif mode == 'overlap':
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
            sr_ratio=cfg.sr_ratios[idx]
         )
         for idx in range(cfg.num_layers)]
    )
    # return_value[0] of build_norm_layer is the name of norm
    norm = build_norm_layer(norm_cfg, total_embed_dims)[1]
    projection_block = ModuleList([patch_embed, encode, norm])
    return projection_block

def random_masking(x, keep_length, ids_shuffle):
    """Apply random masking on input tensor
    Args:
        x: input patches (batch x length x feature)
        keep_length: length of unmasked patches
        ids_shuffle: random indices for shuffling the input sequence 
    Returns:
        kept: unmasked part of x
        mask: a 2D (batch x length) mask tensor of 0s and 1s indicated which
            part of x is masked out. The value 0 indicates not masked and 1
            indicates masked.
        ids_restore: indices to restore x. If we take the kept part and masked
            part of x, concatentate them together and index it with ids_restore,
            we should get x back.
    """
    batch, length, feature_size = x.size()
    shuffled = index_sequence(x, ids_shuffle)
    kept = shuffled[:, :keep_length, :]
    ids_restore = torch.empty_like(ids_shuffle)
    for idx, row in enumerate(ids_shuffle):
        ids_restore[idx] = torch.argsort(row)
    mask = torch.empty(batch, length)
    for i in range(batch):
        mask[i] = torch.where(ids_restore[i] >= keep_length, 1., 0.)
    return kept.cuda(), mask.cuda(), ids_restore

def restore_masked(kept_x, masked_x, ids_restore):
    """Restore masked patches
    Args:
        kept_x: unmasked patches
        masked_x: masked patches
        ids_restore: indices to restore x
    Returns:
        restored patches
    """
    return index_sequence(torch.cat((kept_x, masked_x), dim=1), ids_restore)