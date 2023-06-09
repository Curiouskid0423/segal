### budget configs
DATASET_LEN = 1600 
PIXEL_PER_IMG = int(1138*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
SPG = 2
SAMPLE_ROUNDS = 5
### miscell configs
HEURISTIC = 'mps'
VIZ_SIZE = 25

### path configs
BASE = '../../' 
DATA_FILE = f'{BASE}dataset/cityscapes_src-free.py'
MODEL_FILE = f'{BASE}models/twin_segmenter.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
SCHEDULE = f'{BASE}schedules/default.py'

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
    SCHEDULE
]

### data configs
data = dict( samples_per_gpu=SPG, workers_per_gpu=4 ) 
# custom_imports = dict(
#     imports=[
#         'experiments._base_.dataset_acdc',
#     ], allow_failed_imports=False
# )

# mixed precision
fp16 = dict(loss_scale='dynamic')
# pretrained weights (fcn decoder is random)
load_from = "work_dirs/gtav_init_after50epochs/epoch_50.pth" # mIoU 71.76 on Cityscapes

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
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        loss_decode=dict(
            type='ReconstructionLoss', loss_weight=15.0))
)

# workflow
multitask_validation_iter = 200

optimizer=dict(
    lr=1e-3, 
    weight_decay=5e-5,
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.2),
            'auxiliary_head': dict(lr_mult=1.0),
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

runner = dict(type='MultiTaskActiveRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=300, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)

# workflow and runtime configs
workflow = [
    (('query', 1),),
    (('train', 6), ('query', 1)),
    (('train', 6), ('query', 1)),
    (('train', 6), ('query', 1)),
    (('train', 6), ('query', 1)),
    (('train', 6),)
] # 30 epochs

# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round     = BUDGET, 
            initial_label_pixels = 0,
            sample_evenly        = True, 
            ignore_index         = 255, 
            radius               = 1,
        )),
    visualize = dict(
        size    = VIZ_SIZE, 
        overlay = True,
        dir     = "visualizations/srcfree_ripu-mps_linear_cs_dummy"),
    reset_each_round    = False, 
    heuristic           = HEURISTIC,
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19,
        crop_size=(384, 384),
        expl_schedule='linear'
    )
)

# logger hooks
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs = dict(
        #         entity  = 'syn2real',
        #         project = 'active_domain_adapt',
        #         group   = 'aaai',
        #         name    = f'srcfree_ripu-mps_linear_cs_dummy',
        #     )
        # )
    ]
)
