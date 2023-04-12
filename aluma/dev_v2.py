BASE = '../' 
SPG = 2 # Sample per GPU
SAMPLE_ROUNDS = 5
QUERY_EPOCH = 2
HEURISTICS = "entropy"
BUDGET = int(1280*640*0.01) * 2975

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

_base_ = [
    # './models/segmenter_vit-b16_mask.py', 
    './models/twin_segmenter.py', 
    './dataset/gtav2cityscapes.py',
    f'{BASE}configs/_base_/default_runtime.py', 
    f'./schedules/default.py' 
]

# simplify decoder head to MLP
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

data = dict(samples_per_gpu=SPG, workers_per_gpu=2)
optimizer = dict(lr=0.001, weight_decay=0.0)
log_config = dict(
    interval=10,
    hooks=[ dict(type='TextLoggerHook') ]
)
# ordinary workflow
# workflow = [('train', QUERY_EPOCH), ('query', 1)]
# multitask workflow
mae_warmup_epochs = 10
workflow = [('train', QUERY_EPOCH), ('query', 1)]

runner = dict(
    type='MultiTaskActiveRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
)

# active learning configs
active_learning = dict(
    settings = dict(
        pixel = dict(      
            budget_per_round=BUDGET,           
            initial_label_pixels=BUDGET,
            sample_evenly=True,
            ignore_index=255,
        ),
    ),
    reset_each_round=False,
    heuristic=HEURISTICS,
)
