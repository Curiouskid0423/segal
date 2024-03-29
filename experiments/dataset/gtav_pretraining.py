DATA_ROOT = '/shared/yutengli/data/gtav/pretraining/'
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
scale_size = (1280, 640)
# crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=scale_size, ratio_range=(0.5, 1.5)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=scale_size),
    dict(type='RandomFlip',prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='GTAVDataset',
        data_root=DATA_ROOT,
        img_dir='train',
        ann_dir='labels_train',
        pipeline=train_pipeline),
    query=dict(
        type='GTAVDataset',
        data_root=DATA_ROOT,
        img_dir='train',
        ann_dir='labels_train',
        pipeline=query_pipeline),
    val=dict(
        type='GTAVDataset',
        data_root=DATA_ROOT,
        img_dir='val',
        ann_dir='labels_val',
        pipeline=test_pipeline),
    test=dict(
        type='GTAVDataset',
        data_root=DATA_ROOT,
        img_dir='val',
        ann_dir='labels_val',
        pipeline=test_pipeline)
)