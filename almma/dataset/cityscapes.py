# dataset settings
dataset_type = 'CityscapesDataset' # train=2975, val=500, test=1525
data_root = '/shared/yutengli/data/cityscapes/'
scale_size = (1280, 640) # (width, height) by mmcv convention
crop_size =(384, 384)
source_free = True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=scale_size, ratio_range=(0.8, 2.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'), 
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=scale_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=scale_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=scale_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
        keys=['img'], meta_keys=(
            'filename', 'ori_filename', 'ori_shape', 
            'img_shape', 'scale_factor', 'img_norm_cfg')
    )
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    query=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=query_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/test',
        ann_dir='gtFine/test',
        pipeline=test_pipeline)
    )
