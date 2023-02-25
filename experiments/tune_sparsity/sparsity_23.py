custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

load_from = "experiments/gtv_ckpt_fpnR50.pth"
data = dict(samples_per_gpu=1, workers_per_gpu=2)

_base_ = [ 
    "../../configs/_base_/models/fpn_r50.py", 
    "../dataset/cityscapes.py", 
    "../../configs/_base_/default_runtime.py",
]

active_learning = dict(
    settings = dict(
        pixel = dict(      
            budget_per_round=int(256*512*0.005) * 2975,           
            initial_label_pixels=int(256*512*0.005) * 2975,
            sample_evenly=True,
            ignore_index=255,
        ),
    ),
    visualize = dict(
        size=10,
        overlay=True,
        dir="viz_tune_sparsity_23"
    ),
    reset_each_round=False,
    heuristic="sparsity",
    heuristic_cfg=dict(
        k=2,
        inflection=0.25
    )
)

workflow = [('train', 5), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=10, 
)
evaluation = dict(interval=5, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5*10)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=True)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name="fpnR50_gtav_sparsity_23",
            )
        )
    ]
)