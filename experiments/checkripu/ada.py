# training configs
# QUERY_EPOCH = 4
SAMPLE_ROUNDS = 5
SPG = 1 # Sample per GPU
# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
# path configs
BASE = '../' 
DATA_FILE = f'{BASE}dataset/gtav2cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/deeplabv3plus_r50-d8.py'
# miscell configs
HEURISTIC = "ripu"
VIZ_SIZE = 20

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
load_from = 'experiments/gtav_ckpt_deeplabv3+r101.pth'
data = dict( samples_per_gpu=SPG, workers_per_gpu=4 ) 

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
            initial_label_pixels=0, # target set initially not labelled at all
            sample_evenly=True,
            ignore_index=255,
        ),
    ),
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="visualizations/ada"
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
    heuristic_cfg=dict(
        k=32,
        use_entropy=True,
        categories=19
    )
)

""" ===== Workflow and Runtime configs ===== """
# workflow = [('train', QUERY_EPOCH), ('query', 1)] 
workflow = [
    (('train', 3), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 10),)
] # total of 17 epochs

runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
)
evaluation = dict(interval=1, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real', project='active_domain_adapt', name=f'deeplabv3+r101_ada_ripu'
        #     )
        # )
    ]
)