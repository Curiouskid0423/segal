"""
config file for gtav -> cityscapes on deeplabv3+_resnet101
"""
# NOTE: Using mini GTAV (only 2000 images) for faster iteration
source_free = False
gtav_root = '/shared/yutengli/data/gtav/mini/'
cs_root = '/shared/yutengli/data/cityscapes/'
META_KEYS = (
    'filename', 'mask_filename', 'ori_filename', 
    'ori_shape','img_shape', 'pad_shape', 
    'scale_factor', 'img_norm_cfg')
mask_dir = './work_dirs/ada/masks'
scale_size = (512, 256) # (width, height) by mmcv convention
crop_size =(200, 200)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadMasks', mask_dir=mask_dir),
    dict(type='ResizeWithMask', img_scale=scale_size, ratio_range=(0.8, 2.0)),
    dict(type='RandomFlipWithMask', prob=0.5, direction='horizontal'), 
    dict(type='RandomCropWithMask', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=scale_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'mask'], meta_keys=META_KEYS),
]

query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMasks', mask_dir=mask_dir),
    dict(type='ResizeWithMask', img_scale=scale_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
        keys=['img', 'mask'], 
        meta_keys=META_KEYS,
    )
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=scale_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='GTAVDataset',
        data_root=gtav_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=train_pipeline),   # train on GTAV
    query=dict(
        type='CityscapesDataset',
        data_root=cs_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=query_pipeline),   # query on CS
    val=dict(
        type='CityscapesDataset',
        data_root=cs_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),    # validate on CS
    test=dict(
        type='CityscapesDataset',
        data_root=cs_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))    # validate on CS