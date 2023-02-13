QUERY_EPOCH = 30
QUERY_SIZE = 1310 # (1%) 256 * 512 * 0.01 = 1310.72 pixels
SAMPLE_ROUNDS = 5
GPU = 8
SPG = 1 # Sample per GPU
HEURISTIC = "margin"
MODEL_FILE = '../configs/_base_/models/fpn_r50.py'
DATA_FILE = './dataset/gtav2cityscapes.py'
RUNTIME_FILE = '../configs/_base_/default_runtime.py'

_base_ = [
    MODEL_FILE, 
    DATA_FILE,
    RUNTIME_FILE
]

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)


# Cannot afford workers_per_gpu > 2 on windigo
data = dict( samples_per_gpu=SPG, workers_per_gpu=2 ) 

""" ===== Log configs ===== """
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'(windigo)_fpnr50',
            )
        )
    ]
)

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        pixel = dict(
            sample_threshold=5,              # per image, only sample from top `threshold`%
            query_size=QUERY_SIZE,           # query size (in pixel) at each step 
            initial_label_pixels=QUERY_SIZE, # of pixels labeled randomly at the 1st epoch
            ignore_index=255                 # any value other than 255 fails due to seg_pad_val in Pad transform
        ),
    ),
    # visualize = dict(
    #     size=10,
    #     overlay=True,
    #     dir="gta2cityscapes"
    # ),
    heuristic=HEURISTIC,
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
    pretrained='supervised' # FIXME: allow ['supervised', 'self-supervised', 'random']
)

evaluation = dict(interval=QUERY_EPOCH, by_epoch=True, metric='mIoU', pre_eval=True) # eval on "validation" set
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.)
optimizer_config = dict()