""" SGD (Segmenter) config """
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
# evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=5)

""" AdamW config """
optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.99), weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    # warmup_ratio=1e-3,
    power=1.0,
    # power=0.9, 
    min_lr=1e-5,
    by_epoch=False
)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

####################################
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))