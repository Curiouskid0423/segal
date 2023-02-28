""" mmcv base configs """
BASE = '../' 
DATA_FILE = f'{BASE}dataset/cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/deeplabv3plus_r50-d8.py'
SPG = 1 # Sample per GPU
GPU = 8
""" active learning configs """
QUERY_EPOCH = 4
BUDGET = int(256*512*0.01) * 2975
SAMPLE_ROUNDS = 5
HEURISTIC = "entropy"
VIZ_SIZE = 20

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
load_from = 'experiments/gtav_ckpt_deeplabv3+r101.pth'
data = dict( samples_per_gpu=SPG, workers_per_gpu=1 ) 

# FP16
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=256.)
# fp16 = dict()

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
]

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            budget_per_round=100,
        ),
        # pixel = dict(      
        #     budget_per_round=BUDGET,           
        #     initial_label_pixels=BUDGET,
        #     sample_evenly=True,
        #     ignore_index=255, # any value other than 255 fails
        # ),
    ),
    # visualize = dict(
    #     size=VIZ_SIZE,
    #     overlay=True,
    #     dir="viz_checkripu_ripu_pa_full"
    # ),
    reset_each_round=False,
    heuristic=HEURISTIC,
    # heuristic_cfg=dict(
    #     k=4,
    #     use_entropy=True,
    #     categories=19 # Cityscapes and GTAV have 19 classes
    # )
)

""" ===== Workflow and Runtime configs ===== """
# workflow = [('train', QUERY_EPOCH), ('query', 1)] 
# Total up to 30 epochs (RIPU had 27)
workflow = [
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 12),)
] 
runner = dict(
    type='ActiveLearningRunner', 
    # sample_mode="pixel",  
    sample_mode="image", 
    sample_rounds=SAMPLE_ROUNDS, 
)
evaluation = dict(interval=5, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)
optimizer = dict(type='SGD', lr=0.0009, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'deeplabv3+r101_checkripu_test_image',
            )
        )
    ]
)