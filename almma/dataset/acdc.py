### acdc config
acdc_root = '/shared/yutengli/data/acdc/'
source_free = False
scale_size = (1280, 640) # (W, H) by mmcv convention
crop_size = (384, 384)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

# various pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=scale_size, ratio_range=(0.8, 2.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'), 
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_set_dict = dict(
    type='ACDCDataset', data_root=acdc_root,
    img_dir='rgb_anon/everything/train',
    ann_dir='ground_truth/everything/train',
    pipeline=train_pipeline
)

val_set_dict = dict(
    type='ACDCDataset', data_root=acdc_root,
    img_dir='rgb_anon/everything/val',
    ann_dir='ground_truth/everything/val',
    pipeline=test_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = train_set_dict,
    val = val_set_dict,
    test = val_set_dict
)