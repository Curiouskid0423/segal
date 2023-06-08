### budget configs
SPG = 2
SAMPLE_ROUNDS = 5
DATASET_LEN = 1600 
# PIXEL_PER_IMG = int(1280*640*0.01)
PIXEL_PER_IMG = int(1138*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
### path configs
BASE = '../../' 
DATA_FILE = f'{BASE}dataset/cityscapes2acdc.py'
# MODEL_FILE = f'{BASE}models/twin_segmenter.py'
MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
SCHEDULE = f'{BASE}schedules/default.py'
### miscellaneous configs
HEURISTIC = "ripu"
VIZ_SIZE = 20

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
    SCHEDULE
]

### data configs
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_acdc',
    ], allow_failed_imports=False
)
data = dict( samples_per_gpu=SPG, workers_per_gpu=3 ) 

# mixed precision
# fp16 = dict(loss_scale='dynamic')
# pretrained weights (fcn decoder is random)
load_from = "work_dirs/warmup/iter_1000.pth" 

optimizer=dict(
    lr=1e-3, 
    weight_decay=5e-5,
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.2),
            # 'auxiliary_head': dict(lr_mult=1.0),
        }
    )
)
optimizer_config = dict()

# workflow and runtime configs
workflow = [
    (('train', 4), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 3),)
] # 15 epochs (debug purpose)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1000, 
    power=1.0, 
    min_lr=1e-5,
    by_epoch=False
)

# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round=BUDGET, initial_label_pixels=0,
            sample_evenly=True, ignore_index=255, radius=1,
        )),
    visualize = dict(
        size=VIZ_SIZE, overlay=True,
        dir="visualizations/n2a_mps_r5_ripu_no-mtl_baseline"),
    reset_each_round = False, 
    heuristic = HEURISTIC,
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19 
    )
)

# logger hooks
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs = dict(
                entity  = 'syn2real',
                project = 'active_domain_adapt',
                group   = 'aaai',
                name    = f'n2a_mps_r5_ripu_no-mtl_baseline',
            )
        )
    ]
)

runner = dict(type='ActiveLearningRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=300, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)