""" mmcv base configs """
BASE = './' 
DATA_FILE = f'{BASE}dataset/cityscapes.py'
RUNTIME_FILE = f'{BASE}../configs/_base_/default_runtime.py'
MODEL_FILE = f'{BASE}../configs/_base_/models/fpn_r50.py'
# SCHEDULE_FILE = f'{BASE}../configs/_base_/schedules/schedule_40k.py'
SPG = 2 # Sample per GPU
GPU = 4
""" active learning configs """
QUERY_EPOCH = 2 
QUERY_SIZE = int(256*512*0.01)
SAMPLE_ROUNDS = 10
HEURISTIC = "entropy"

custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

# model = dict(
#     init_cfg=dict(
#         type='Pretrained',
#         checkpoint='experiments/gtv_ckpt_fpnR50.pth',
#     )
# )
# work_dir = 'work_dir'

resume_from = 'experiments/gtv_ckpt_fpnR50.pth'

_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
]

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            query_size=10
        ),
        pixel = dict(      
            # sample_threshold=5,
            query_size=QUERY_SIZE,           
            initial_label_pixels=QUERY_SIZE, # of pixels labeled randomly at the 1st epoch
            sample_evenly=True,              # FIXME: ignored for the current development phase
            ignore_index=255                 # any value other than 255 fails due to seg_pad_val in Pad transform
        ),
    ),
    # visualize = dict(
    #     size=VIZ_SIZE,
    #     overlay=True,
    #     dir="viz_test_cfg"
    # ),
    heuristic=HEURISTIC,
    shuffle_prop=0.0,                        # FIXME: ignored for the current development phase
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
)
evaluation = dict(interval=QUERY_EPOCH//2, by_epoch=False, metric='mIoU', pre_eval=True)
checkpoint_config = dict(by_epoch=True, interval=QUERY_EPOCH)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=True)
# optimizer = dict(type='Adam', lr=1e-4, weight_decay=0.) 
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(
        #     type='WandbLoggerHookWithVal',
        #     init_kwargs=dict(
        #         entity='syn2real',
        #         project='active_domain_adapt',
        #         name=f'fpn50_r50_gtav_DevRun',
        #     )
        # )
    ]
)