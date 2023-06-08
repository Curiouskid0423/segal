### budget configs
DATASET_LEN = 1600 
PIXEL_PER_IMG = int(1280*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
### miscellaneous configs
HEURISTIC = "random"
VIZ_SIZE = 20

_base_ = [ './_base_.py' ]

optimizer=dict(
    paramwise_cfg = dict(
        custom_keys={ 
            'backbone': dict(lr_mult=0.4),
            'auxiliary_head': dict(lr_mult=0.4),
        }
    )
)

# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round=BUDGET, initial_label_pixels=0,
            sample_evenly=True, ignore_index=255, radius=1,
        )),
    visualize = dict(
        size=VIZ_SIZE, overlay=True,
        dir="visualizations/n2a_lr-mult-enc04_warmup1000"),
    reset_each_round=False, heuristic=HEURISTIC)

# logger hooks
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs = dict(
                entity  = 'syn2real',
                project = 'active_domain_adapt',
                group   = 'aaai',
                name    = f'n2a_lr-mult-enc04_warmup1000',
            )
        )
    ]
)
