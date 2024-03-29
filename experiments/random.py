# training configs
QUERY_EPOCH = 1
SPG = 2
SAMPLE_ROUNDS = 5
# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(256*512*0.01)
BUDGET = PIXEL_PER_IMG * DATASET_LEN
# path configs
BASE = './' 
DATA_FILE = f'{BASE}dataset/cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/deeplabv3plus_r50-d8.py'
# miscellaneous configs
HEURISTIC = "random"
VIZ_SIZE = 20

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
load_from = 'experiments/gtav_ckpt_deeplabv3+r101.pth'
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 

# mixed precision
fp16 = dict(loss_scale='dynamic')

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
]

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
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="visualizations/viz_random"
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
)
evaluation = dict(interval=QUERY_EPOCH//2, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
optimizer = dict(type='SGD', lr=0.00075, momentum=0.9, weight_decay=0.0005)
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
                name=f'deeplabv3+r101_pixel_random',
            )
        )
    ]
)