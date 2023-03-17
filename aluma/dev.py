BASE = '../' 
SPG = 1 # Sample per GPU
GPU = 8
SAMPLE_ROUNDS = 3
# QUERY_EPOCH = 1
HEURISTICS = "entropy"
BUDGET = int(256*512*0.01) * 2975

_base_ = [
    './models/twin_mit.py', # './models/nonoverlap_twin_mit.py',
    f'{BASE}experiments/dataset/cityscapes.py',
    f'{BASE}configs/_base_/default_runtime.py', 
    f'./schedules/default.py' 
]

# just a stronger model (~ mit-b2)
model = dict(
    backbone=dict(
        seg_projection=dict(embed_dims=64), 
        mae_projection=dict(embed_dims=64),
        encoder=dict(embed_dims=64, num_layers=[4, 6, 3])),
    decode_head=dict( in_channels=[64, 128, 320, 512] ),
    auxiliary_head=dict( in_channels=[64, 128, 320, 512] ),
)


data = dict(samples_per_gpu=SPG, workers_per_gpu=2)

# workflow and runtime configs
mae_warmup_epochs = 150
workflow = [((('train_seg', 4), ('train_mae', 2)), 3), ('query', 1)]
runner = dict(
    type='MultiTaskActiveRunner', 
    sample_mode="pixel", sample_rounds=SAMPLE_ROUNDS)

# 4iters * 8gpus * 8images = 256images as the effective batch size
optimizer = dict( lr = 0.00002, betas=(0.9, 0.95) )
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=4)

log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real',
        #         project='active_domain_adapt',
        #         name=f'debug_warmup-run_multitask-segformer_2080Ti',
        #     )
        # )
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
