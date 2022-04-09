# Plan of Development

BAAL has any basic building blocks already. Read the code and adapt it 
such that MMSeg backbones may use seamlessly. The goal of this baseline system
is to enable the wide variety of MMSeg backabones to perform active learning.

- Expected result
  - Running your baseline system should involve similar procedure to running MMSeg, meaning that
  users should be able to simply change the config and run everything. Files to modify includes: 
  all Dataset-related files, make train_segmentor iteratively label the unlabeled-pool as in 
  `ActiveLearningLoop.step()`.
  - `bash <script> <config_file> <GPU_number>`

- ActiveLearningDataset
  - In baal, this is simply a 1-layer wrapper around a `torchdata.Dataset` class.
  - In MMSeg, the call stack from main() to constructing a dataloader is the following:
  `train_segmentor` > `build_dataloader` (which takes in a torchdata.Dataset and returns a torch.DataLoader,
  the Dataset class is passed in at the train_segmentor call.)
  - Stick with "image" level acquisition for now, but your system's value should be in 
  (1) enabling "pixel" or "region" level acquisition and (2) integration with MMSeg.
- Heursitics: add `AL-RIPU` with your own implementation

- Implement `ReAL` and test if it works on semantic segmentation!

----------------

wrapper = ModelWrapper(model, ...)
>> `wrapper.model` is of type "nn.Module". Pass this into the typical MMSeg pipeline.
al_dataset = ActiveLearningDataset(dataset, test_transform = {...})
>> `al_dataset.dataset` should be a torchdata.Subset that you can specify index with.
Simply append new index on it every time.


- Take in all the arguments
- Run (customized) train_segmentor
- Inside train_segmentor, we cannot define a static dataloader at the beginning because our we will be modifying the Dataset class (specifically, via appending more indices onto the indices list passed into torchdata.Subset), thus every epoch will need to instantiate a new DataLoader with our updated torchdata.Subset


If we want to avoid changes in MMCV, we might have to: for every epoch, call `runner.run(<static dataloader>, cfg.workflow)` with the currently-available labeled dataloader; somehow keep the state_dict (might not always need it but need to have this implemented); update the indices of torchdata.Subset and create a corresponding DataLoader; pass this new DataLoader to runner to initiate another epoch.



```
2022-04-09 20:46:12,442 - mmseg - INFO - Config:
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dilation=6),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        dilation=6),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/fcn_d6_r50-d16_512x1024_40k_cityscapes'
gpu_ids = range(0, 4)
auto_resume = False

```

```
Config keys:
norm_cfg
model
dataset_type
data_root
img_norm_cfg
crop_size
train_pipeline
test_pipeline
data
log_config
dist_params
log_level
load_from
resume_from
workflow
cudnn_benchmark
optimizer
optimizer_config
lr_config
runner
checkpoint_config
evaluation
work_dir
gpu_ids
auto_resume
seed
norm_cfg
model
dataset_type
data_root
img_norm_cfg
crop_size
train_pipeline
test_pipeline
data
log_config
dist_params
log_level
load_from
resume_from
workflow
cudnn_benchmark
optimizer
optimizer_config
lr_config
runner
checkpoint_config
evaluation
work_dir
gpu_ids
auto_resume
seed
norm_cfg
model
dataset_type
data_root
img_norm_cfg
crop_size
train_pipeline
test_pipeline
data
log_config
dist_params
log_level
load_from
resume_from
workflow
cudnn_benchmark
optimizer
optimizer_config
lr_config
runner
checkpoint_config
evaluation
work_dir
gpu_ids
auto_resume
seed
```