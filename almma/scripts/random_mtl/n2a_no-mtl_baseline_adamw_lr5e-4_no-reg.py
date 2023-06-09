### budget configs
DATASET_LEN = 1600
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
SPG = 2
SAMPLE_ROUNDS = 5

### path configs
BASE = '../../' 
DATA_FILE = f'{BASE}dataset/cityscapes2acdc.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
SCHEDULE = f'{BASE}schedules/default.py' 

# miscellaneous configs
HEURISTIC = "random"
VIZ_SIZE = 20

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
    SCHEDULE
]

# data configs
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_acdc',
    ], allow_failed_imports=False
)

data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 
# mixed precision
# fp16 = dict(loss_scale='dynamic')
# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round=BUDGET,           
            initial_label_pixels=0,
            sample_evenly=True,
            ignore_index=255,
            radius=1,
        ),
    ),
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="visualizations/n2a_no-mtl_baseline_adamw_lr5e-4"
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)

# workflow and runtime configs
workflow = [
    (('train', 4), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 3),)
] # 15 epochs (debug purpose)

load_from = "work_dirs/gtav_init_after50epochs/epoch_50.pth" # mIoU 71.76 on Cityscapes
runner = dict(type='ActiveLearningRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=300, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=10)

# optimizers and learning rate
optimizer=dict(
    lr=5e-4, 
    weight_decay=0.,
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.2),
            # 'auxiliary_head': dict(lr_mult=0.2),
        }
    )
)
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1000, 
    power=1.0, 
    min_lr=1e-5,
    by_epoch=False
)


# logger hooks
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity  = 'syn2real',
                project = 'active_domain_adapt',
                group   = 'aaai',
                name    = f'ada_no-mtl_baseline_adamw_cs2acdc_lr5e-4',
            )
        )
    ]
)
