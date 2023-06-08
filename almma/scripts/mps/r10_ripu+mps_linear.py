### budget configs
DATASET_LEN = 1600 
# PIXEL_PER_IMG = int(1280*640*0.01)
PIXEL_PER_IMG = int(1138*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
### miscellaneous configs
HEURISTIC = "mps"
VIZ_SIZE = 20

_base_ = [ './_base_.py' ]
load_from = "work_dirs/gtav_init_after50epochs/epoch_50.pth" # 71.76 (~20k iter)

# workflow and runtime configs
workflow = [(('train', 2), ('query', 1)) for _ in range(10)] + [(('train', 3),)]
runner = dict(type='MultiTaskActiveRunner', sample_mode="region", sample_rounds=10)


# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round=BUDGET, initial_label_pixels=0,
            sample_evenly=True, ignore_index=255, radius=1,
        )),
    visualize = dict(
        size=VIZ_SIZE, overlay=True,
        dir="visualizations/n2a_mps_r10_ripu+mps_linear_strong-init"),
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
                name    = f'n2a_mps_r10_ripu+mps_linear_strong-init',
            )
        )
    ]
)
