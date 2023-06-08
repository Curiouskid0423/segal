### budget configs
DATASET_LEN = 2975 
PIXEL_PER_IMG = int(1280*640*1.) 
BUDGET = PIXEL_PER_IMG * DATASET_LEN 
### miscell configs
HEURISTIC = 'random'
VIZ_SIZE = 20

_base_ = ['./_base_.py']

load_from = "work_dirs/gtav_init/epoch_50.pth" # mean IoU: 70.39

# workflow and runtime configs
workflow = [('train', 50), ('query', 1)] 

optimizer=dict( lr=1e-4 )
lr_config = dict(
    _delete_=True,
    policy='poly',
    # warmup='linear',
    # warmup_iters=1000, 
    power=1.0, 
    min_lr=1e-5,
    by_epoch=False
)

# active learning configs
active_learning = dict(
    settings = dict(
        region = dict(      
            budget_per_round    = 0, 
            initial_label_pixels = BUDGET,
            sample_evenly       = True, 
            ignore_index        = 255, 
            radius              = 1,
        )),
    visualize = dict(
        size=VIZ_SIZE, overlay=True,
        dir="visualizations/long_warmup_gtav_init_after50epochs"),
    reset_each_round = False, 
    heuristic = HEURISTIC,
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
                name    = f'long_warmup_gtav_init_after50epochs',
            )
        )
    ]
)
