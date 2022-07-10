_base_ = [
    '../configs/_base_/models/fcn_hr18.py', '../configs/_base_/datasets/cityscapes.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_20k.py'
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real',
        #         project='al_baseline',
        #         name='fcn_hr18_gpu4_e48_q100x4_entropy_dev',
        #     )
        # )
    ]
)

# Reference formula: num_worker = 4 * num_GPU 
# data = dict(samples_per_gpu=16, workers_per_gpu=4)    # 1gpu
# data = dict(samples_per_gpu=2, workers_per_gpu=4)     # 8gpu

active_learning = dict(
    initial_pool=100, query_size=100, heuristic="entropy",
    shuffle_prop=0.0, query_epoch=1,
    )

workflow = [('train', 1)]
runner = dict(type='ActiveLearningRunner', max_epochs=24, max_iters=None)
checkpoint_config = dict(by_epoch=True, interval=8)
evaluation = dict(interval=8, by_epoch=True, metric='mIoU', pre_eval=True)

# FIXME:
# lr_config = dict(policy='poly', power=0.9**{GPU_NUM / query_epoch}, min_lr=1e-4, by_epoch=True)
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)