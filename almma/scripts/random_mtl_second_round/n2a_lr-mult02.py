_base_ = [ '../random_mtl/n2a_lr1e-3_wd5e-5_recloss15.py' ]

lr_config = dict(_delete_=True, policy='poly', power=1.0, min_lr=1e-5, by_epoch=False)

# logger hooks
log_config = dict(
    hooks=[
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs = dict(
                entity  = 'syn2real',
                project = 'active_domain_adapt',
                group   = 'aaai',
                name    = f'n2a_lr-mult-enc02',
            )
        )
    ]
)
