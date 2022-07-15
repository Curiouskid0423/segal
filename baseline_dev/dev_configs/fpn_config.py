_base_ = [
    '../../configs/_base_/models/fpn_r50.py', 
    './cityscapes_pixels.py',
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
                name='fpn_r50_pix_gpu4_e20_q20x1_entropy',
            )
        )
    ]
)

""" ===== Active Learning configs ===== """
active_learning = dict(
    sample_mode="image",
    image_based_settings=dict(
        initial_pool=100, 
        query_size=100
        ),
    pixel_based_settings=dict(
        budget=400, 
        # budget: the number of pixels sampled from "each" image. 
        # will sample "evenly" from each image.
        sample_threshold=5, 
        # for each image, only sample from top `sample_threshold` percentage
        query_size=20, 
        # query size (in pixel) at each step. e.g. 100 pixels at a step
        sample_evenly=True, # FIXME: ignored for the current development phase.
        # sampling pixels evenly across each image yields much better results
        ignore_index=255, 
        # ignore_index: set ignore_index according to the dataset (NOTE: any 
        # value other than 255 bugs due to seg_pad_val in Pad transform. Fix this.) 
        initial_label_pixels=20
        # initial_label_pixels: number of pixels labelled randomly at 
        # the first epoch (before any sampling)
        ),
    heuristic="entropy",
    query_epoch=1,
    shuffle_prop=0.0,
    )

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', 1)]
runner = dict(type='ActiveLearningRunner', max_epochs=20, max_iters=None)
checkpoint_config = dict(by_epoch=True, interval=8)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# lr_config = dict(policy='poly', power=0.9**{GPU_NUM / query_epoch}, min_lr=1e-4, by_epoch=True)
optimizer = dict(type='Adam', lr=0.00001, weight_decay=0.00005)
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
evaluation = dict(interval=1, by_epoch=True, metric='mIoU', pre_eval=True)

# Reference formula: num_worker = 4 * num_GPU 