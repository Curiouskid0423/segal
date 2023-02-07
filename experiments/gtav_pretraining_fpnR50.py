DATA_FILE = './dataset/gtav.py'
RUNTIME_FILE = '../configs/_base_/default_runtime.py'
MODEL_FILE = '../configs/_base_/models/fpn_r50.py'
SCHEDULE_FILE = '../configs/_base_/schedules/schedule_40k.py'
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
                name=f'fpn50_r50_gtav_itr40k_batch{SPG*GPU}_IN-initialized',
            )
        )
    ]
)