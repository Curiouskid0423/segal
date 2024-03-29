# _base_ = '../../configs/_base_/models/segmenter_vit-b16_mask.py'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'  # noqa
# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
IMG_SIZE = (384, 384)
STRIDE = (256, 256)
model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=IMG_SIZE,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    # decode_head=dict(
    #     type='SegmenterMaskTransformerHead',
    #     in_channels=768,
    #     channels=768,
    #     num_classes=150,
    #     num_layers=2,
    #     num_heads=12,
    #     embed_dims=768,
    #     dropout_ratio=0.0,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    # ),
    decode_head=dict(
        type='FCNHead',
        in_channels=768,
        channels=768,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=IMG_SIZE, stride=STRIDE),
)