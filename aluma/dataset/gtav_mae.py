# dataset configs
# gtav_root = '/shared/yutengli/data/gtav/mini/' 
gtav_root = '/shared/yutengli/data/gtav/'
cs_root = '/shared/yutengli/data/cityscapes/'
source_free = True
scale_size = (1280, 640) # (width, height) by mmcv convention
crop_size = (384, 384)   # (512, 512)
mask_dir = './work_dirs/warmup/masks'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# various pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadMasks', mask_dir=mask_dir),
    dict(type='ResizeWithMask', img_scale=scale_size, ratio_range=(0.8, 2.0)),
    dict(type='RandomFlipWithMask', prob=0.5, direction='horizontal'), 
    dict(type='RandomCropWithMask', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=scale_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=scale_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
        keys=['img'],
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

source_dataset_dict = dict(
    type='GTAVDataset', data_root=gtav_root, 
    img_dir='images', ann_dir='labels')
target_dataset_dict = dict(
    type='CityscapesDataset', data_root=cs_root,
    img_dir='leftImg8bit/train', ann_dir='gtFine/train')
val_dataset_dict = dict(
    type='CityscapesDataset', data_root=cs_root,
    img_dir='leftImg8bit/val', ann_dir='gtFine/val')
train_source_dataset = dict(**source_dataset_dict, pipeline=train_pipeline)
train_target_dataset = dict(**target_dataset_dict, pipeline=train_pipeline)
query_dataset = dict(**target_dataset_dict, pipeline=query_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = train_source_dataset,
    query = query_dataset,
    val = dict(**val_dataset_dict, pipeline=test_pipeline),
    test = dict(**val_dataset_dict, pipeline=test_pipeline)
)