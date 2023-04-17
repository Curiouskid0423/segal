# How to edit Segal configs


A config file in Segal is essentially an extension from the standard `mmseg` config file with an additional `active_learning: dict` item and two types of customized runner instances, `ActiveLearningRunner` and `MultiTaskActiveRunner`. Below are the functionalities for each of the arguments of Segal, and some critical features in `mmseg`.

```
""" GTAV DATASET IMPORT """
custom_imports = dict(
    imports=[
        'experiments._base_.dataset_gtav',
    ], allow_failed_imports=False
)

""" PRETRAINED WEIGHTS """
load_from = 'experiments/model.pth'

""" BASE MMCV CONFIGS """
_base_ = [ 
    MODEL_FILE, 
    DATA_FILE, 
    RUNTIME_FILE,
    SCHEDULE_FILE
]

""" ACTIVE LEARNING CONFIGS """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            budget_per_round=10
        ),
        pixel = dict(      
            sample_threshold=5,
            budget_per_round=BUDGET,           
            initial_label_pixels=BUDGET,
            sample_evenly=True,
            ignore_index=255
        ),
    ),
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="viz_folder"
    ),
    reset_each_round=True,
    heuristic=HEURISTIC,
    heuristic_cfg=dict(
        k=1,
        use_entropy=True,
        categories=19 # Cityscapes and GTAV have 19 classes
    )
)

""" WORKFLOW (two options)"""
workflow = [('train', QUERY_EPOCH), ('query', 1)]   # option 1 (regular)
workflow = [                                        # option 2 (irregular)
    (('train', 3), ('query', 1)),
    (('train', 1), ('query', 1)),
    (('train', 10),)] 

""" RUNNER CONFIGS (two options) """
runner = dict(type='ActiveLearningRunner', sample_mode="pixel", sample_rounds=5)    # option 1
runner = dict(type='MultiTaskActiveRunner', sample_mode="pixel", sample_rounds=5)   # option 2

""" MultiTaskActiveRunner specific settings """
mask_dir = './work_dirs/my_config_name/masks'
mae_viz_dir = 'mae_warmup_images' # can be omitted

""" LOGGER CONFIGS """
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHookWithVal',
            init_kwargs=dict(
                entity='syn2real',
                project='<project_name>',
                name='<experiment_name>'))
    ]
)

```
- `custom_imports`: This is a MMCV feature that allows users to import customized module imports. In this example, we are adding GTAV Dataset class.
- `load_from`: A MMCV feature to load pretrained weights. In this example, we load in a GTAV-pretrained FPN-ResNet50.
- `active_learning`
    - `settings`: Contains either image- or pixel-based sampling configs. User may leave both image- and pixel-based settings in the config, but only one will be used according to the `sample_mode` specified in `runner` config.
        - `image`: Contains all image-based sampling configs. `initial_pool` denotes the number of labelled images available at the beginning of training. `budget_per_round` denotes the number of images queried per sampling round. 
        - `pixel`: Contains all pixel-based sampling config, designed according to the [PixelPick](https://github.com/NoelShin/PixelPick) paper. `sample_threshold` denotes top K% of uncertainty scores per image from which we sample. This can be removed or set to False if you prefer sampling exactly the top ranked entries deterministically. `budget_per_round` denotes the number of total pixels sampled per round. For example, if we sample on average 1% per image, our `budget_per_round` may be:<br> 0.01 * (256 * 512) * 2975 = 1310 * 2975 = 3,897,250 pixels. <br> `initial_label_pixels` denotes the initially available pixels per image. This quantity is typically the same as query_size in standard AL, but 100% in Active Domain Adaptation. `ignore_index=255` is a masking value to ignore the gradients comptued from those unlabelled pixels, do not change it to any other number. 
    - `visualize`: Contains settings to visualize the map of sampled pixels. `size` denotes number of visualized images/maps exported. `overlay` is a binary value to indicate whether to overlay the labelled pixels (shown as white dots) over the input image. `dir` specifies the folder to store visualizations in.
    - `reset_each_round`: A binary variable to determine whether to reset the weights after each round of sampling is performed.
    - `heuristics`: Defines the sampling heuristic function.
    - `heuristics_cfg`: Currently only supported when using `RIPU` heuristics, as illustrated in the example above. Defines the hyperparameters in the desired heuristic function. 
- `workflow`: The workflow config works the same way as the standard `mmseg` configs, but allows a **query** tuple after the **train** tuple. There are two cases:
    - **sample regularly**
    When sampling regularly, an example workflow will be `[('train', K), ('query', 1)]`. This means train for K epochs after each round of sampling. Effectively, the total epochs over the entire training will be `K * sample_rounds`. In the above pseudocode, the model queries for new labels every QUERY_EPOCH epoch.  
    - **sample irregularly**
    Oftentimes it is more optimal to sample irregularly. For instance, in RIPU, when performing region-based sampling they ran 40k iterations in total, but sample at `[10k, 12k, 14k, 18k, 20k]`. To sample irregularly, ie. at manually defined epoch, replace `workflow` according to this example: 
        ```
        workflow = [
            (('train', 10), ('query', 1)),
            (('train', 2),  ('query', 1)),
            (('train', 2),  ('query', 1)),
            (('train', 2),  ('query', 1)),
            (('train', 14),)
        ] 
        ```
        In this example, there are 5 sample rounds (considering the `initial_pool`) and the total epochs is 30 but allocated heavily at the beginning and the end. Note that in the last round `(('train', 14),)`, that **LAST comma** is not omittable.
    
- `runner`
    - `type`: Use the customized runner `ActiveLearningRunner` in Segal, which allows iteratively adding new labels into the dataloader.
    - `sample_mode`: Define the sampling mode, either "pixel" or "image". Will also allow region-based sampling in the near future. 
    - `sample_rounds`: Numbers of sampling rounds. For example, setting this to 5 means sampling `query_size` amount of labels for five times.