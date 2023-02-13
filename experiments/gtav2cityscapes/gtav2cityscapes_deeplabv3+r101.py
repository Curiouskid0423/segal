QUERY_EPOCH = 5
QUERY_SIZE = 1310   # 256*512 * 0.01 = 1310 pixels
SAMPLE_ROUNDS = 5
GPU = 2
SPG = 1             # Sample per GPU
HEURISTIC = "margin"
MODEL_FILE = '../configs/_base_/models/deeplabv3plus_r50-d8.py'
DATA_FILE = './dataset/gtav2cityscapes.py' 
RUNTIME_FILE = '../configs/_base_/default_runtime.py'

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

# Cannot afford workers_per_gpu > 2 on Windigo
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 

""" ===== Log configs ===== """
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real',
        #         project='active_domain_adapt',
        #         name=f'(Titan)_r101_round{SAMPLE_ROUNDS}_q1%_gpu{GPU}',
        #     )
        # )
    ]
)

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        # image = dict(
        #     initial_pool=100, 
        #     query_size=10
        # ),
        pixel = dict(
            sample_threshold=5,              
            query_size=QUERY_SIZE,          
            initial_label_pixels=QUERY_SIZE,
            ignore_index=255                
        ),
    ),
    # visualize = dict(
    #     size=10,
    #     overlay=True,
    #     dir="active_learn_fpnR50"
    # ),
    heuristic=HEURISTIC,
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS
)

evaluation = dict(interval=QUERY_EPOCH, by_epoch=True, metric='mIoU', pre_eval=True) 
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.)
optimizer_config = dict()