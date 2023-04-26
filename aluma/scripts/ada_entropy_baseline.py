# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN
SPG = 2
SAMPLE_ROUNDS = 1
# path configs
BASE = '../' 
DATA_FILE = f'{BASE}dataset/gtav2cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
# miscellaneous configs
HEURISTIC = "entropy"
VIZ_SIZE = 20

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
]

# data configs
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 
# mixed precision
fp16 = dict(loss_scale='dynamic')
# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round=BUDGET,           
            initial_label_pixels=0,
            sample_evenly=True,
            ignore_index=255,
            radius=1,
        ),
    ),
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="visualizations/ada_entropy_baseline"
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)

# workflow and runtime configs
workflow = [
    (('train', 1), ('query', 1)),
    (('train', 1), ('query', 1)),
] 

runner = dict(type='ActiveLearningRunner', sample_mode="region", sample_rounds=SAMPLE_ROUNDS)
evaluation = dict(interval=500, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)

# optimizers and learning rate
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)

# logger hooks
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'ada_entr_vit-b_16_ra',
            )
        )
    ]
)

###############################################################################
# # budget configs
# DATASET_LEN = 2975
# PIXEL_PER_IMG = int(1280*640*0.01) 
# BUDGET = PIXEL_PER_IMG * DATASET_LEN
# SPG = 2
# SAMPLE_ROUNDS = 5
# # path configs
# BASE = '../' 
# DATA_FILE = f'{BASE}dataset/gtav2cityscapes.py'
# RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
# MODEL_FILE = f'{BASE}models/segmenter_linear_vit-b16.py'
# # miscellaneous configs
# HEURISTIC = "entropy"
# VIZ_SIZE = 20

# _base_ = [ 
#     MODEL_FILE, 
#     DATA_FILE, 
#     RUNTIME_FILE,
# ]

# # data configs
# custom_imports = dict(
#     imports=[
#         'experiments._base_.dataset_gtav',
#     ], allow_failed_imports=False
# )
# data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 
# mask_dir = './work_dirs/ada_entropy_baseline/masks'
# # mixed precision
# fp16 = dict(loss_scale='dynamic')
# # active learning configs
# active_learning = dict(
#     settings = dict(
#         pixel = dict(      
#             budget_per_round=BUDGET,           
#             initial_label_pixels=0,
#             sample_evenly=True,
#             ignore_index=255,
#         ),
#     ),
#     visualize = dict(
#         size=VIZ_SIZE,
#         overlay=True,
#         dir="visualizations/ada_entropy_baseline"
#     ),
#     reset_each_round=False,
#     heuristic=HEURISTIC,
# )

# # workflow and runtime configs
# workflow = [
#     (('train', 2), ('query', 1)),
#     (('train', 1), ('query', 1)),
#     (('train', 1), ('query', 1)),
#     (('train', 1), ('query', 1)),
#     (('train', 1), ('query', 1)),
#     (('train', 10),)
# ] # total of 17 epochs

# runner = dict(type='ActiveLearningRunner', sample_mode="pixel", sample_rounds=SAMPLE_ROUNDS)
# evaluation = dict(interval=500, by_epoch=False, metric='mIoU', pre_eval=True)
# checkpoint_config = dict(by_epoch=True, interval=5)

# # optimizers and learning rate
# optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)

# # logger hooks
# log_config = dict(
#     interval=40,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(
#             type='WandbLoggerHookWithVal',
#             init_kwargs=dict(
#                 entity='syn2real', project='active_domain_adapt', 
#                 name=f'segmenter-linear_vit-b_16_entropy_baseline'
#             )
#         )
#     ]
# )