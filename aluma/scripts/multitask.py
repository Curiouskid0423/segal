BASE = '../' 
SPG = 2
SAMPLE_ROUNDS = 5
QUERY_EPOCH = 2
HEURISTICS = "entropy"
BUDGET = int(1280*640*0.01) * 2975

# base modules
_base_ = [
    f'{BASE}models/twin_segmenter.py', 
    f'{BASE}dataset/gtav2cityscapes.py',
    f'{BASE}../configs/_base_/default_runtime.py', 
    f'{BASE}schedules/default.py' 
]

# data configs
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False)
data = dict(samples_per_gpu=SPG, workers_per_gpu=4)
evaluation = dict(interval=200, by_epoch=False)

load_from = "work_dirs/warmup/epoch_10.pth"
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
fp16 = dict(loss_scale='dynamic')

# workflow
optimizer = dict(lr=0.0005, weight_decay=0.) #weight_decay=5e-4)
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    power=1.0, 
    min_lr=1e-5,
    by_epoch=False)

log_config = dict(
    interval=40,
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'aluma_vit-b_16_mae-IN1k-init_multitask',
            )
        )
    ]
)

# ordinary workflow
# workflow = [('train', QUERY_EPOCH), ('query', 1)]
# multitask workflow
workflow = [
    (('train', 3), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 10),)
]

runner = dict(
    type='MultiTaskActiveRunner', sample_mode="pixel", sample_rounds=SAMPLE_ROUNDS)

# active learning configs
active_learning = dict(
    settings = dict(
        pixel = dict(      
            budget_per_round=BUDGET,           
            initial_label_pixels=0,
            sample_evenly=True,
            ignore_index=255,
        ),
    ),
    reset_each_round=False,
    heuristic=HEURISTICS,
)
