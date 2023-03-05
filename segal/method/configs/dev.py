BASE = '../../' 
SPG = 2 # Sample per GPU
GPU = 8

_base_ = [
    './twin_mit.py',
    f'{BASE}../experiments/dataset/cityscapes.py',
    f'{BASE}../configs/_base_/default_runtime.py', 
    f'{BASE}../configs/_base_/schedules/schedule_20k.py'
]

""" ===== Workflow and Runtime configs ===== """
# train for a total of 10 epochs before sampling
# (mae -> seg * 4) * 2 --> query

# workflow = [
#     (
#         ((('mae', 1), ('seg', 4)), 2), 
#         ('query', 1)
#     )
# ] 
# runner = dict(
#     type='MultitaskActiveRunner', 
#     sample_mode="pixel",
#     sample_rounds=5, 
# )

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)