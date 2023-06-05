################################################################
# this config models most of the settings from multitask_e26 in order to be a baseline.
# control factors: lr, lr_config, batch, optimizer, weight_init
#                  al_config, workflow
################################################################
# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
SPG = 2
SAMPLE_ROUNDS = 5
# path configs
BASE = '../' 
DATA_FILE = f'{BASE}dataset/gtav2cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
SCHEDULE = f'{BASE}schedules/default.py' 
# miscellaneous configs
HEURISTIC = "ripu"
VIZ_SIZE = 20

# base modules
_base_ = [
    MODEL_FILE,
    DATA_FILE,
    RUNTIME_FILE, 
    SCHEDULE
]
# data configs
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)
data = dict(samples_per_gpu=SPG, workers_per_gpu=4)
# mixed precision
# fp16 = dict(loss_scale='dynamic')
load_from = "work_dirs/warmup/iter_1000.pth" # fcn decoder is random
# workflow
optimizer=dict(
    # _delete_=True,
    lr=5e-4, 
    weight_decay=5e-5,
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.1),
        }
    )
)

# lr config
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500, # 500
    power=1.0, 
    min_lr=1e-5,
    by_epoch=False
)

log_config = dict(
    interval=40,
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'ada_ripu_vit-b16_gtav-mae-init_ra_batch16_1500warmup_e26',
            )
        )
    ]
)

workflow = [
    (('train', 6), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 8),)
]
runner = dict(type='ActiveLearningRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=500, by_epoch=False)
checkpoint_config = dict(by_epoch=True, interval=5)

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
        dir="visualizations/ada_ripu_vit-b16_gtav-mae-init_ra_batch16_1500warmup_e26"
    ),
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19 
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)