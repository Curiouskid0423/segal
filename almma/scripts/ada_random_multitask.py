# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
SPG = 2
SAMPLE_ROUNDS = 5
# path configs
BASE = '../' 
DATA_FILE = f'{BASE}dataset/gtav10K2cityscapes.py'
MODEL_FILE = f'{BASE}models/twin_segmenter.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
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
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 

# mixed precision
# fp16 = dict(loss_scale='dynamic')
# pretrained weights
load_from = "work_dirs/warmup/iter_1000.pth" # fcn decoder is random
# model configs: simplify decoder head to mlp
model = dict(
    decode_head=dict(
        _delete_=True,
        type='FCNHead',
        in_channels=768,
        channels=768,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
)

# workflow
multitask_validation_iter = 200
optimizer=dict(
    # _delete_=True,
    lr=7.5e-4, 
    weight_decay=5e-5,
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.2),
            'auxiliary_head': dict(lr_mult=0.2),
        }
    )
)
# optimizer=dict(
#     # _delete_=True,
#     lr=7.5e-4, 
#     weight_decay=5e-5,
#     paramwise_cfg = dict(
#         custom_keys={ 
#             'backbone': dict(lr_mult=0.2),
#             'auxiliary_head': dict(lr_mult=0.2),
#         }
#     )
# )

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1000, 
    power=1.0, 
    min_lr=2e-5,
    by_epoch=False
)

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
        dir="visualizations/ada_random_multitask"
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)

# workflow and runtime configs
workflow = [
    (('train', 5), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 5),)
] # 22 epochs
runner = dict(type='MultiTaskActiveRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=500, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)
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
                name=f'almma_random_vit-b16_ra_gtav10K',
            )
        )
    ]
)
