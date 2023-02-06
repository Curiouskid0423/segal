# How to edit Segal configs


A config file in Segal is essentially an extension from the standard `mmseg` config file with an additional `active_learning: dict` item and a customized runner instance, `ActiveLearningRunner`. Below are the functionalities for each of the arguments:

```

""" ===== Active Learning configs ===== """
active_learning = dict(
    settings = dict(
        image = dict(
            initial_pool=100, 
            query_size=10
        ),
        pixel = dict(
            sample_threshold=5,              
            query_size=QUERY_SIZE,           
            initial_label_pixels=QUERY_SIZE, 
            ignore_index=255 
        ),
    ),
    visualize = dict(
        size=VIZ_SIZE,
        overlay=True,
        dir="viz_reproduce_fix"
    ),
    heuristic=HEURISTIC
)

""" ===== Workflow and Runtime configs ===== """
workflow = [('train', QUERY_EPOCH), ('query', 1)] 
runner = dict(
    type='ActiveLearningRunner', 
    sample_mode="pixel", 
    sample_rounds=SAMPLE_ROUNDS, 
    pretrained='supervised' # FIXME: allow ['supervised', 'self-supervised', 'random']
)
```

- `active_learning`
    - `settings`: Contains either image- or pixel-based sampling configs. User may leave both image- and pixel-based settings in the config, but only one will be used according to the `sample_mode` specified in `runner` config.
        - `image`: Contains all image-based sampling configs. `initial_pool` denotes the number of labelled images available at the beginning of training. `query_size` denotes the number of images queried per sampling round. 
        - `pixel`: Contains all pixel-based sampling config, designed according to the [PixelPick](https://github.com/NoelShin/PixelPick) paper. `sample_threshold` denotes top K% of uncertainty scores per image from which we sample. `query_size` denotes the number pixels sampled per round. `initial_label_pixels` denotes the initially available pixels per image. This quantity is typically the same as query_size in standard AL, but 100% in Active Domain Adaptation. `ignore_index=255` is a masking value to ignore the gradients comptued from those unlabelled pixels, do not change it to any other number. 
    - `visualize`: Contains settings to visualize the map of sampled pixels. `size` denotes number of visualized images/maps exported. `overlay` is a binary value to indicate whether to overlay the labelled pixels (shown as white dots) over the input image. `dir` specifies the folder to store visualizations in.
    - `heuristics`: Defines the sampling heuristic function.
- `workflow`: The workflow config works the same way as the standard `mmseg` configs, but with an additional **query** tuple after the "train" tuple. In the above example, the model queries for new labels every QUERY_EPOCH epoch.  
- `runner`
    - `type`: Use the customized runner `ActiveLearningRunner` in Segal, which allows iteratively adding new labels into the dataloader.
    - `sample_mode`: Define the sampling mode, either "pixel" or "image". Will also allow region-based sampling in the near future. 
    - `sample_rounds`: Numbers of sampling rounds. For example, setting this to 5 means sampling `query_size` amount of labels for five times.
    - `pretrained`: Set weight initialization. Currently only "supervised" is valid. Should have three options ('supervised', 'self-supervised', 'random']) after bug fix.