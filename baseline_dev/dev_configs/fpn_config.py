_base_ = [
    '../../configs/_base_/models/fpn_r50.py', 
    # Change this to Cityscapes file in OpenMIM if running image-based sampling
    './dataset/cityscapes_pixel.py',        
    '../../configs/_base_/default_runtime.py'
]

""" ===== Log configs ===== """
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='al_baseline',
                name='fpn-r50_pix_gpu8_a10_q10_e50_margin',
            )
        )
    ]
)

""" ===== Active Learning configs ===== """
QUERY_EPOCH = 40
QUERY_SIZE = 10
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            query_size=100
        ),
        pixel = dict(
            sample_threshold=5, 
            # for each image, only sample from top `sample_threshold` percentage
            query_size=QUERY_SIZE, 
            # query size (in pixel) at each step. e.g. 100 pixels at a step
            sample_evenly=True, # FIXME: ignored for the current development phase.
            # sampling pixels evenly across each image yields much better results
            ignore_index=255, 
            # ignore_index: set ignore_index according to the dataset (NOTE: any 
            # value other than 255 bugs due to seg_pad_val in Pad transform. Fix this.) 
            initial_label_pixels=QUERY_SIZE
            # initial_label_pixels: number of pixels labelled randomly at 
            # the first epoch (before any sampling)
        )
    ),
    heuristic="margin",
    shuffle_prop=0.0, # FIXME: ignored at the current phase.
    )

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', 1)]
runner = dict(type='ActiveLearningRunner', sample_mode="pixel", sample_rounds=10, query_epochs=QUERY_EPOCH)
evaluation = dict(interval=QUERY_EPOCH, by_epoch=True, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
optimizer = dict(type='Adam', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
