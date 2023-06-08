# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
SPG = 2
SAMPLE_ROUNDS = 5
# path configs
BASE = '../' 
MODEL_FILE = f'{BASE}models/twin_segmenter.py'
DATA_FILE = f'{BASE}dataset/gtav2cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
SCHEDULE = f'{BASE}schedules/default.py' 
# miscellaneous configs
HEURISTICS = "ripu"
VIZ_SIZE = 20
work_dir = './work_dirs/multitask_ra_lr5e-4_m50_1500warmup_e26'

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
    ], allow_failed_imports=False)
data = dict(samples_per_gpu=SPG, workers_per_gpu=4)
evaluation = dict(interval=500, by_epoch=False)

# load_from = "/home/yutengli/workspace/vit_pretrained_checkpoints/mae_visualize_vit_base_mmcv.pth"
load_from = "work_dirs/warmup/iter_1000.pth" # fcn decoder is random
multitask_validation_iter = 200

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

# mixed precision
# fp16 = dict(loss_scale='dynamic')

# workflow
optimizer=dict(
    # _delete_=True,
    lr=5e-4, 
    weight_decay=5e-5, #0.5
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.1),
            'auxiliary_head': dict(lr_mult=0.1),
        }
    )
)

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
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real',
        #         project='active_domain_adapt',
        #         name=f'almma_vit-b16_mae-IN1k-init_multitask_ra_batch16_1500warmup_e26',
        #     )
        # )
    ]
)

# ordinary workflow
# workflow = [('train', QUERY_EPOCH), ('query', 1)]
# multitask workflow
# workflow = [
#     (('train', 5), ('query', 1)),
#     (('train', 3), ('query', 1)),
#     (('train', 3), ('query', 1)),
#     (('train', 2), ('query', 1)),
#     (('train', 2), ('query', 1)),
#     (('train', 8),)
# ] # e23
workflow = [
    (('train', 6), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 3), ('query', 1)),
    (('train', 50),)
]

runner = dict(type='MultiTaskActiveRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)

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
        dir="visualizations/multitask_ra_lr5e-4_m50_1500warmup_e26"
    ),
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19 
    ),
    reset_each_round=False,
    heuristic=HEURISTICS,
)