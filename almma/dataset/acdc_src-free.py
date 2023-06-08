################################################
# For MTL warmup (both MAE and Segmentation)   #
################################################

# dataset configs
acdc_root = '/shared/yutengli/data/acdc/'
source_free = True
scale_size = (1138, 640) # (width, height) by mmcv convention
crop_size = (384, 384)   # (512, 512)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)
META_KEYS = ('filename', 'mask_filename', 'ori_filename', 'ori_shape',
             'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')

# various pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadMasks'),
    dict(type='ResizeWithMask', img_scale=scale_size, ratio_range=(0.8, 2.0)),
    dict(type='RandomFlipWithMask', prob=0.5, direction='horizontal'), 
    dict(type='RandomCropWithMask', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=scale_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'mask'], meta_keys=META_KEYS),
]
query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMasks'),
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

train_source_dataset = dict(
    type='ACDCDataset', data_root=acdc_root,
    img_dir='rgb_anon/everything/train',
    ann_dir='ground_truth/everything/train',
    pipeline=train_pipeline)

val_dataset_dict = dict(
    type='ACDCDataset', data_root=acdc_root,
    img_dir='rgb_anon/everything/val',
    ann_dir='ground_truth/everything/val')

query_dataset = dict(
    type='ACDCDataset', data_root=acdc_root,
    img_dir='rgb_anon/everything/train',
    ann_dir='ground_truth/everything/train', 
    pipeline=query_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = train_source_dataset,
    query = query_dataset,
    val = dict(**val_dataset_dict, pipeline=test_pipeline),
    test = dict(**val_dataset_dict, pipeline=test_pipeline)
)