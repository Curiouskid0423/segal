import sys

_base_ = [
    '../configs/_base_/models/fcn_r50-d8.py', '../configs/_base_/datasets/cityscapes.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(dilation=6),
    auxiliary_head=dict(dilation=6)
    )


active_learning = dict(
    initial_pool=100,
    query_size=100,
    heuristic="random",
    )