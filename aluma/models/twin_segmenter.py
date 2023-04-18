# checkpoints
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'
# checkpoint =  "../mae_pretrain_vit_base_mmcv.pth"
checkpoint = "/home/yutengli/workspace/vit_pretrained_checkpoints/mae_visualize_vit_base_mmcv.pth"

# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
# POS_EMBED_TYPE = 'fixed' # sinusoidal position embedding
POS_EMBED_TYPE = 'learnable'

# IMG_SIZE = (512, 512)
IMG_SIZE = (384, 384)
# IMG_SIZE = (224, 224)

# EVAL_STRIDE = (480, 480)
EVAL_STRIDE = (300, 300)
# EVAL_STRIDE = (150, 150)

model = dict(
    type='MultiTaskSegmentor',
    # pretrained=checkpoint,
    pretrained=None,
    mae_config=dict(
        mask_ratio=0.75,
    ),
    backbone=dict(
        type='SharedVisionTransformer',
        img_size=IMG_SIZE,
        patch_size=16,
        in_channels=3,
        pos_embed_type=POS_EMBED_TYPE,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        with_cls_token=False,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels=768,
        num_classes=19,
        num_layers=2,
        num_heads=12,
        embed_dims=768,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type='MaskDecodeHead',
        img_size=IMG_SIZE,
        patch_size=16,
        in_channels=768,
        channels=512,
        embed_dims=512,
        num_layers=8,
        num_heads=16,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        num_classes=3,
        pos_embed_type=POS_EMBED_TYPE,
        norm_cfg=backbone_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ReconstructionLoss', loss_weight=0.3),
    ),
    test_cfg=dict(mode='slide', crop_size=IMG_SIZE, stride=EVAL_STRIDE),
)