# mixed precision training (sometimes result in NaN loss)
# fp16 = dict(loss_scale='dynamic')
# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)
optimizer_config = dict()
# learning policy
lr_config = dict(
    # _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
evaluation = dict(interval=1, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=10)