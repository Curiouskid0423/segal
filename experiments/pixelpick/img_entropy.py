# training configs
QUERY_EPOCH = 25 # 50 epochs
SAMPLE_ROUNDS = 10
SPG = 2 # sample per gpu
# path configs
BASE = '../' 
MODEL_FILE = f'{BASE}../configs/_base_/models/fpn_r50.py'
DATA_FILE = f'{BASE}dataset/cityscapes_no_mask.py' 
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
# miscell configs
HEURISTIC = "entropy"

_base_ = [
    MODEL_FILE, 
    DATA_FILE,
    RUNTIME_FILE
]


""" ===== Log configs ===== """
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(entity='syn2real', project='active_domain_adapt', name=f'img_entropy_e10')
        )
    ]
)

data = dict( samples_per_gpu=SPG, workers_per_gpu=4 )

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            budget_per_round=100
        ),
    ),
    reset_each_round=False,
    heuristic=HEURISTIC,
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="image", 
    sample_rounds=SAMPLE_ROUNDS)

evaluation = dict(interval=QUERY_EPOCH//2, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=1e-4)
optimizer_config = dict()