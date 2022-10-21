QUERY_EPOCH = 40 # PixelPick setting is 50 epochs
QUERY_SIZE = 10
SAMPLE_ROUNDS = 10
GPU = 8
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
                name=f'(rescale)_fpn-r50_pix_gpu{GPU}_a{SAMPLE_ROUNDS}_q{QUERY_SIZE}_e{QUERY_EPOCH}_{HEURISTIC}',
            )
        )
    ]
)

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            query_size=10
        ),
        pixel = dict(
            sample_threshold=5,         # per image, only sample from top `threshold`%
            query_size=QUERY_SIZE,      # query size (in pixel) at each step 
            initial_label_pixels=QUERY_SIZE, # of pixels labeled randomly at the 1st epoch
            sample_evenly=True,         # FIXME: ignored for the current development phase
            # sampling pixels evenly across each image yields much better results
            ignore_index=255            # set ignore_index according to the # of classes 
            # (FIXME: any value other than 255 bugs due to seg_pad_val in Pad transform.)
        ),
    ),
    heuristic=HEURISTIC,
    shuffle_prop=0.0, # FIXME: ignored at the current phase.
    )

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
    pretrained='supervised' # FIXME: allow ['supervised', 'self-supervised', 'random']
    ) 
# runner = dict(
#     type='ActiveLearningRunner', sample_mode="image", sample_rounds=1, query_epochs=QUERY_EPOCH)
evaluation = dict(interval=QUERY_EPOCH//4, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
optimizer = dict(type='Adam', lr=3e-4, weight_decay=1e-4) # 12:20AM 10/21/2022
optimizer_config = dict()
