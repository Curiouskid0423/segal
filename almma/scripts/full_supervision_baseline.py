"""
A fully supervised upper baseline for Segmenter-Linear
reproduced on the same runner instance.
"""

# useful variables
SPG = 2
BASE = '../' 
DATA_FILE = f'{BASE}dataset/cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
SCHEDULE_FILE = f'{BASE}../configs/_base_/schedules/schedule_160k.py'
# base mmcv config
_base_ = [
    DATA_FILE, 
    MODEL_FILE,
    RUNTIME_FILE,
    SCHEDULE_FILE
]
# data configs
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 
# mixed precision and optimizer
fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1000, by_epoch=False, metric='mIoU', pre_eval=True)
optimizer = dict(lr=0.001, weight_decay=0.0)
# logger hooks
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real', project='active_domain_adapt', 
        #         name=f'segmenter-linear_vit-b16_full_supervision_cs'
        #     )
        # )
    ]
)