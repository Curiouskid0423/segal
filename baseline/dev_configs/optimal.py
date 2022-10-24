QUERY_EPOCH = 40 # PixelPick setting is 50 epochs
QUERY_SIZE = 10
SAMPLE_ROUNDS = 10
GPU = 2
SPG = 2 # sample per gpu
HEURISTIC = "margin"
MODEL_FILE = '../../configs/_base_/models/fpn_r50.py'
DATA_FILE = './dataset/cityscapes_pixel.py' 
RUNTIME_FILE = '../../configs/_base_/default_runtime.py'
# Original CityScapes dataset: '../../configs/_base_/datasets/

_base_ = [
    MODEL_FILE, 
    DATA_FILE,
    RUNTIME_FILE
]


""" ===== Log configs ===== """
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='al_baseline',
                name=f'(V100)_fpn-r50_pix_bth{GPU*SPG}_act{SAMPLE_ROUNDS}_qry{QUERY_SIZE}_e{QUERY_EPOCH}_lr1e-4_wd0_{HEURISTIC}',
            )
        )
    ]
)
data = dict( samples_per_gpu=SPG, workers_per_gpu=4 )

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            query_size=10
        ),
        pixel = dict(
            initial_label_pixels=QUERY_SIZE,
            sample_threshold=5,
            query_size=QUERY_SIZE,
            sample_evenly=True,     # FIXME: ignored for the current development phase
            ignore_index=255            
        ),
    ),
    heuristic=HEURISTIC,
    shuffle_prop=0.0,               # FIXME: ignored for the current development phase
    )

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
    pretrained='supervised' # FIXME: allow ['supervised', 'self-supervised', 'random']
    )
evaluation = dict(interval=QUERY_EPOCH//5, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.) 
optimizer_config = dict()