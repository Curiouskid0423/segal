# training configs
QUERY_EPOCH = 1 # 50 epochs
QUERY_SIZE = 10
SAMPLE_ROUNDS = 10
SPG = 1 # sample per gpu
# budget configs
DATASET_LEN = 2975
PIXEL_PER_IMG = 10
BUDGET = PIXEL_PER_IMG * DATASET_LEN # int(256*512*0.01) * 2975
# path configs
BASE = '../' 
MODEL_FILE = f'{BASE}../configs/_base_/models/fpn_r50.py'
DATA_FILE = f'{BASE}dataset/cityscapes.py' 
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
# miscell configs
HEURISTIC = "margin"
VIZ_SIZE = 10

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
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real', project='active_domain_adapt', name=f'pixelpick_e20_b8')
        # )
    ]
)

data = dict( samples_per_gpu=SPG, workers_per_gpu=4 )

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        pixel = dict(      
            sample_threshold=5,             # per image, only sample from top `threshold`%
            budget_per_round=BUDGET,        # query size (in pixel) at each step 
            initial_label_pixels=BUDGET,    # of pixels labeled randomly at the 1st epoch
            sample_evenly=True,
            ignore_index=255                # any value other than 255 fails due to seg_pad_val in Pad transform
        ),
    ),
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="visualizations/pixelpick"
    ),
    reset_each_round=True,
    heuristic=HEURISTIC,
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS)
# evaluation = dict(interval=QUERY_EPOCH//5, by_epoch=False, metric='mIoU', pre_eval=True)
evaluation = dict(interval=QUERY_EPOCH, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
optimizer = dict(type='Adam', lr=2e-4, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001,)
optimizer_config = dict()