BASE = '../../' 
DATA_FILE = f'{BASE}dataset/gtav.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/deeplabv3plus_r50-d8.py'
SCHEDULE_FILE = f'{BASE}../configs/_base_/schedules/schedule_80k.py'
SPG = 1 # Sample per GPU
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

# model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
data = dict(samples_per_gpu=SPG, workers_per_gpu=2) 

# optimizer
# optimizer = dict(type='Adam', lr=3e-4, weight_decay=0.)
# optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)

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
                name=f'deeplabv3+_r50_gtav_itr80k_batch{SPG*GPU}_IN-initialized',
            )
        )
    ]
)