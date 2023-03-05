_base_ = ['./b0_pretraining.py']
SPG = 1 # Sample per GPU
GPU = 8

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 8, 27, 3]
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512]))


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='active_domain_adapt',
                name=f'segformer_b4_batch{SPG*GPU}_512x512_2080Ti',
            )
        )
    ]
)