custom_imports = dict(
    imports=[
        'dataset_gtav',
        'hook_wandb',
    ], allow_failed_imports=False)

_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/gtav2cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
# data = dict(samples_per_gpu=16, workers_per_gpu=4) # 1gpu
# data = dict(samples_per_gpu=4, workers_per_gpu=4) # 4gpu
data = dict(samples_per_gpu=2, workers_per_gpu=4) # 8gpu
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='gtav2cityscapes',
                name='fcn_hr18_512x1024_20k_gtav2cityscapes',
            )
        )
    ]
)
