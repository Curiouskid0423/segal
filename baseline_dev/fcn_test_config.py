_base_ = [
    '../configs/_base_/models/fcn_hr18.py', '../configs/_base_/datasets/cityscapes.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_20k.py'
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='al_baseline',
                name='fcn_hr18_gpu8_e12_q100x1',
            )
        )
    ]
)

active_learning = dict(
    initial_pool=100, query_size=100, heuristic="random",
    shuffle_prop=0.0, iterations=20, learning_epoch=1,
    query_epoch=1,
    )
workflow = [('train', 1)]
runner = dict(type='ActiveLearningRunner', max_epochs=12, max_iters=None)
lr_config = dict(policy='poly', power=0.8, min_lr=1e-4, by_epoch=True)
evaluation = dict(interval=4, by_epoch=True, metric='mIoU', pre_eval=True)