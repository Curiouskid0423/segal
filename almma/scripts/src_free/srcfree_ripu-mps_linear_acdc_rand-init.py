### budget configs
DATASET_LEN = 1600 
PIXEL_PER_IMG = int(1138*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
### miscell configs
HEURISTIC = 'mps'
VIZ_SIZE = 20
_base_ = ['./_base_.py']

### data configs
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_acdc',
    ], allow_failed_imports=False
)

# workflow and runtime configs
runner = dict( sample_rounds=4 )
workflow = [
    (('train', 6), ('query', 1)),
    (('train', 6), ('query', 1)),
    (('train', 6), ('query', 1)),
    (('train', 6), ('query', 1)),
    (('train', 6),)
] # 30 epochs

# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round     = BUDGET, 
            initial_label_pixels = BUDGET,
            sample_evenly        = True, 
            ignore_index         = 255, 
            radius               = 1,
        )),
    visualize = dict(
        size=VIZ_SIZE, overlay=True,
        dir="visualizations/srcfree_ripu-mps_linear_acdc_smallstride_rand-init"),
    reset_each_round = False, 
    heuristic = HEURISTIC,
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19,
        crop_size=(384, 384),
        expl_schedule='linear'
    )
)

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
                name    = f'srcfree_ripu-mps_linear_acdc_smallstride_rand-init',
            )
        )
    ]
)
