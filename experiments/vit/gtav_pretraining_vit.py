BASE = '../' 
DATA_FILE = f'{BASE}dataset/gtav.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/upernet_vit-b16_ln_mln.py'
SCHEDULE_FILE = f'{BASE}../configs/_base_/schedules/schedule_160k.py'
SPG = 2 # Sample per GPU
GPU = 8

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

_base_ = [
    MODEL_FILE,
    DATA_FILE,
    RUNTIME_FILE,
    SCHEDULE_FILE
]
data = dict(samples_per_gpu=SPG, workers_per_gpu=2)

model = dict(
    pretrained='experiments/vit/ckpts/vit-base-p16_in1k-224.pth',
    backbone=dict(drop_path_rate=0.1, final_norm=True),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150)
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=SPG, workers_per_gpu=2) 

evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'upernet_vit-b16_gtav_itr160k_batch{SPG*GPU}_BlockerRun',
            )
        )
    ]
)
