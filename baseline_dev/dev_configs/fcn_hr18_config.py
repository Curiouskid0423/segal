_base_ = [
    '../configs/_base_/models/fcn_hr18.py', '../configs/_base_/datasets/cityscapes.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_20k.py'
]

log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real',
        #         project='al_baseline',
        #         name='fcn_hr18_pix_gpu4_e10_q100x1_entropy',
        #     )
        # )
    ]
)

active_learning = dict(
    sample_mode="pixel",
    image_based_settings=dict(
        initial_pool=100, 
        query_size=100
        ),
    pixel_based_settings=dict(
        budget=1000, 
        # budget: the number of pixels sampled from "each" image. 
        # will sample "evenly" from each image.
        sample_threshold=5, 
        # for each image, only sample from top `sample_threshold` percentage
        query_size=100, 
        # query size (in pixel) at each step. e.g. 100 pixels at a step
        sample_evenly=True, # FIXME: ignored for the current development phase.
        # sampling pixels evenly across each image yields much better results
        ignore_index=255, 
        # ignore_index: set ignore_index according to the dataset (FIXME: any 
        # value other than 255 bugs due to seg_pad_val in Pad transform. Fix this.) 
        initial_label_pixels=100
        # initial_label_pixels: number of pixels labelled randomly at 
        # the first epoch (before any sampling)
        ),
    heuristic="random",
    query_epoch=1,
    shuffle_prop=0.0,
    )

workflow = [('train', 1)]
runner = dict(type='ActiveLearningRunner', max_epochs=20, max_iters=None)
checkpoint_config = dict(by_epoch=True, interval=8)
evaluation = dict(interval=3, by_epoch=False, metric='mIoU', pre_eval=True)

# FIXME:
# lr_config = dict(policy='poly', power=0.9**{GPU_NUM / query_epoch}, min_lr=1e-4, by_epoch=True)
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

# Reference formula: num_worker = 4 * num_GPU 
# data = dict(samples_per_gpu=16, workers_per_gpu=4)    # 1gpu
# data = dict(samples_per_gpu=2, workers_per_gpu=4)     # 8gpu