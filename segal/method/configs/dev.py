BASE = '../../' 
SPG = 1 # Sample per GPU
GPU = 8
SAMPLE_ROUNDS = 5
# QUERY_EPOCH = 1
HEURISTICS = "entropy"
BUDGET = int(256*512*0.01) * 2975

_base_ = [
    './twin_mit.py', #'./nonoverlapping_twin_mit.py',
    f'{BASE}../experiments/dataset/cityscapes.py',
    f'{BASE}../configs/_base_/default_runtime.py', 
]

data = dict(samples_per_gpu=SPG, workers_per_gpu=2)

""" ===== Workflow and Runtime configs ===== """
mae_warmup_epochs = 10 # run mae for 10 warmup epochs
workflow = [((('train_seg', 4), ('train_mae', 3)), 5), ('query', 1)]
runner = dict(
    type='MultiTaskActiveRunner', 
    sample_mode="pixel",
    sample_rounds=SAMPLE_ROUNDS, 
)

fp16 = dict(loss_scale='dynamic') # mixed precision (NaN loss sometimes)
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.0001, # lr=0.00006,
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
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

evaluation = dict(interval=1, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=10)
optimizer_config = dict()

log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'debug_warmup-run_multitask-segformer_2080Ti',
            )
        )
    ]
)

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        pixel = dict(      
            budget_per_round=BUDGET,           
            initial_label_pixels=BUDGET,
            sample_evenly=True,
            ignore_index=255, # any value other than 255 fails due to seg_pad_val in Pad transform
        ),
    ),
    reset_each_round=False,
    heuristic=HEURISTICS,
)
