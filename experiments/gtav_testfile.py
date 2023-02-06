DATA_FILE = './dataset/gtav2cityscapes.py'
RUNTIME_FILE = '../configs/_base_/default_runtime.py'
MODEL_FILE = '../configs/_base_/models/segformer_mit-b0.py'
SPG = 1 # Sample per GPU

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

_base_ = [
    MODEL_FILE,
    DATA_FILE,
    RUNTIME_FILE
]

data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 

# optimizer
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(interval=2, metric='mIoU', pre_eval=True)


log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'run_gta2cityscapes',
            )
        )
    ]
)