### budget configs
DATASET_LEN = 1600 
# PIXEL_PER_IMG = int(1280*640*0.01)
PIXEL_PER_IMG = int(1138*640*0.01) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
### miscellaneous configs
HEURISTIC = "ripu"
VIZ_SIZE = 20

_base_ = [ './_base_.py' ]
# load_from = "work_dirs/gtav_init/epoch_50.pth" # mean IoU: 70.39
load_from = "work_dirs/gtav_init_after50epochs/epoch_50.pth" # 71.76 (~20k iter)

# workflow and runtime configs
workflow = [
    (('train', 4), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 2), ('query', 1)),
    (('train', 3),)
] # 15 epochs (debug purpose)

# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round=BUDGET, initial_label_pixels=0,
            sample_evenly=True, ignore_index=255, radius=1,
        )),
    visualize = dict(
        size=VIZ_SIZE, overlay=True,
        dir="visualizations/n2a_mps_r5_ripu_baseline_strong-init"),
    reset_each_round = False, 
    heuristic = HEURISTIC,
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19 
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
                name    = f'n2a_mps_r5_ripu_baseline_strong-init',
            )
        )
    ]
)
