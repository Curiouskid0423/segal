### budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
SPG = 2
SAMPLE_ROUNDS = 5
### path configs
BASE = '../' 
DATA_FILE = f'{BASE}dataset/cityscapes2acdc.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
# effective lr
BASE_LR = 2.5e-4  # official repo
# BASE_LR = 1.5e-4
BASE_BATCH = 2 
# miscellaneous configs
HEURISTIC = "random"
VIZ_SIZE = 20

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
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
        dir="visualizations/n2a_ada_random_baseline"
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)

# workflow and runtime configs
# workflow = [
#     (('train', 5), ('query', 1)),
#     (('train', 3), ('query', 1)),
#     (('train', 3), ('query', 1)),
#     (('train', 3), ('query', 1)),
#     (('train', 3), ('query', 1)),
#     (('train', 5),)
# ] # 22 epochs
workflow = [
    (('train', 6), ('query', 1)),
    (('train', 4), ('query', 1)),
    (('train', 4), ('query', 1)),
    (('train', 4), ('query', 1)),
    (('train', 4), ('query', 1)),
    (('train', 6),)
] # 28 epochs (debug purpose)

runner = dict(type='ActiveLearningRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=300, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=10)

# optimizers and learning rate
optimizer = dict(
    type='SGD', 
    lr=BASE_LR * (SPG * 8) / BASE_BATCH, 
    momentum=0.9, 
    weight_decay=0.0005,
)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

# logger hooks
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'ada_random_vit-b16_ra_cs2acdc',
            )
        )
    ]
)
