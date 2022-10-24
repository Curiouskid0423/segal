TOTAL_EPOCH = 50
GPU = 2
SPG = 2 # Sample per GPU
MODEL_FILE = '../../configs/_base_/models/fpn_r50.py'
DATA_FILE = './dataset/cityscapes_pixel.py' 
# DATA_FILE = '../../configs/_base_/datasets/cityscapes.py' # Original CityScapes dataset
RUNTIME_FILE = '../../configs/_base_/default_runtime.py'

_base_ = [
    MODEL_FILE, 
    DATA_FILE,
    RUNTIME_FILE
]

# Cannot afford workers_per_gpu > 2 on windigo
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 

""" ===== Log configs ===== """
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='al_baseline',
                name=f'(Titan)_fpn-r50_fully-sup_bth{GPU*SPG}_e{TOTAL_EPOCH}_lr3e-4_wd1e-6',
            )
        )
    ]
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', 1)] 
runner = dict( type='EpochBasedRunner', max_epochs=TOTAL_EPOCH)
evaluation = dict(interval=TOTAL_EPOCH//25, by_epoch=True, metric='mIoU', pre_eval=True) # eval on "validation" set
checkpoint_config = dict(by_epoch=True, interval=TOTAL_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
optimizer = dict(type='Adam', lr=3e-4, weight_decay=1e-6) 
optimizer_config = dict()