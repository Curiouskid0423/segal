BASE = '../' 
SPG = 8
GPU = 8
ACCUM_ITER = 2
HEURISTICS = "entropy"
BUDGET = int(1280*640*0.01) * 2975
# BASE_LR = 1.5e-4
BASE_LR = 6e-5

_base_ = [
    f'{BASE}models/twin_segmenter.py', 
    f'{BASE}dataset/cityscapes_mae.py',
    f'{BASE}../configs/_base_/default_runtime.py', 
    f'{BASE}schedules/default.py' 
]

# load in the full model (encoder and decoder)
load_from = "/shared/yutengli/vit_ckpts/warmup/mae_visualize_vit_base_mmcv.pth"
mae_viz_dir = 'mae_warmup_cityscapes_fixed'
# data configs
data = dict(samples_per_gpu=SPG, workers_per_gpu=4) # 12*8=96 images
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
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# mixed precision
fp16 = dict(loss_scale='dynamic')
optimizer = dict(lr=BASE_LR*(SPG*GPU*ACCUM_ITER/256), betas=(0.9, 0.95), weight_decay=5e-3)
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=ACCUM_ITER)

# logger configs
log_config = dict(
    interval=40,
    hooks=[ 
        dict(type='TextLoggerHook'), 
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'mae_batch64x2_warmup_cs_fixed',
            )
        )
    ]
)

# workflow configs
mae_warmup_epochs = 45
mae_val_iter = 500
workflow = [('train', mae_warmup_epochs)]
runner = dict(type='MultiTaskActiveRunner', sample_mode="pixel", warmup_only=True)
evaluation = dict(interval=mae_warmup_epochs, metric='mIoU', pre_eval=True)
checkpoint_config = dict(_delete_=True, by_epoch=False, interval=500)

# active learning configs (just a placeholder for mae warmup)
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
