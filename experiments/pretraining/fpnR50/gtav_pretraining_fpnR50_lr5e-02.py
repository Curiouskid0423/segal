BASE = '../../' 
DATA_FILE = f'{BASE}dataset/gtav.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/fpn_r50.py'
SCHEDULE_FILE = f'{BASE}../configs/_base_/schedules/schedule_40k.py'

SPG = 2 # Sample per GPU
GPU = 4

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

# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=5e-4, by_epoch=False)

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
                name=f'fpn50_r50_gtav_itr40k_batch{SPG*GPU}_IN-initialized_lr5e-02',
            )
        )
    ]
)