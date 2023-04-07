norm_cfg = dict(type='SyncBN', requires_grad=True)

# check out details of EncoderDecoder at mmseg Github /mmseg/models/segmentors/encoder_decoder.py
model = dict(
    type='MultiTaskSegmentor',
    pretrained=None,
    backbone=dict(
        type='TwinMixVisionTransformer',
        seg_projection=dict(
            in_channels=3, embed_dims=32, num_layers=3, 
            num_heads=1, patch_size=7, stride=4, sr_ratios=8,
        ),
        mae_projection=dict(
            in_channels=3, embed_dims=32, num_layers=3,
            num_heads=1, patch_size=8, sr_ratios=1,
            mask_ratio=0.3, rec_crop_size=(160, 160)
        ),
        encoder=dict(
            embed_dims=32,
            num_stages=3,
            num_layers=[2, 2, 2],
            num_heads=[2, 5, 8],
            patch_sizes=[3, 3, 3],  # fix patch_size at each layer. downsample by sr_ratio.
            strides=[2, 2, 2],      # only used in seg stage for PatchEmbedding.
            sr_ratios=[4, 2, 1],    # spatial_reduction used in Attention. ignored in MAE stage.
            out_indices=(0, 1, 2),
        ),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # auxiliary_head is customized to be trained in alternation. 
    auxiliary_head=dict(
        type='MaskDecodeHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3, # predicting RGB value of an image
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='ReconstructionLoss', loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))