QUERY_EPOCH = 1 # PixelPick setting is 50 epochs
# Try training with iteratively increasing epochs: 240 = 16 * (1+5)*5/2
# Split into 250 = (16+10) + 32 + 48 + 64 + 80
QUERY_SIZE = 10 # 1300 pixels = 1% of 256x512
SAMPLE_ROUNDS = 10
GPU = 2
SPG = 2 # Sample per GPU
HEURISTIC = "margin"
MODEL_FILE = '../../configs/_base_/models/fpn_r50.py'
DATA_FILE = './dataset/cityscapes_pixel.py' 
RUNTIME_FILE = '../../configs/_base_/default_runtime.py'
# Original Cityscapes dataset: '../../configs/_base_/datasets/

_base_ = [
    MODEL_FILE, 
    DATA_FILE,
    RUNTIME_FILE
]

# Cannot afford workers_per_gpu > 2 on Windigo
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
        #         project='al_baseline',
        #         name=f'(Titan)_fpn-r50_pix_bth{GPU*SPG}_act{SAMPLE_ROUNDS}_qry1%_e{QUERY_EPOCH}_lr2e-4_wd0_{HEURISTIC}_viz',
        #     )
        # )
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
            sample_threshold=5,              # per image, only sample from top `threshold`%
            query_size=QUERY_SIZE,           # query size (in pixel) at each step 
            initial_label_pixels=QUERY_SIZE, # of pixels labeled randomly at the 1st epoch
            ignore_index=255                 # any value other than 255 fails due to seg_pad_val in Pad transform
        ),
    ),
    visualize = dict(
        size=10,
        overlay=True,
        dir="viz_reproduce_fix"
    ),
    heuristic=HEURISTIC,
    shuffle_prop=0.0, # from BAAL package. ignored at the current phase.
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