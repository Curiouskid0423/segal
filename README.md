# Active Learning Baseline System 

- **2022.09.02** Resolved multiple bugs: (1) weights re-init at each round (2) lr scheduler restart at each round (3) test pipeline applied during sampling instead of train pipeline (4) resolved pixel truncation non-deterministic behavior. Action Items in the coming 3 days:
    - Enable visualization of the sampled pixels at each round on a segmentation map.
    - Refactor as needed to accommodate the case where we gradually increase *BOTH labeled pixels and images.*
    - Add documentation to essential functions and "how to use config file".
- **2022.05.09.** Resolved deadlock issue for distributed training. 
- **2022.05.02.** Current system works with single GPU training, as well as two heuristics options -- random and mean-entropy. Distributed training mode contains some memory issue within step() function to be fixed.

#### Getting started
- Create and edit a config file (the way you interact with MMLab modules). Follow the example in `dev_configs/fpn_config.py`. The only differences from conventional usages of OpenMMLab are the `active_learning` and `runner` configs.
    - `runner = dict(type='ActiveLearningRunner', sample_mode="pixel", sample_rounds=10, query_epochs=QUERY_EPOCH)`

- Train with command:`bash baseline_dev/dist_train_al.sh <config_file> <gpu_number>`

#### Key files
```
baseline_dev/
|-- active
|   |-- __init__.py
|   |-- active_loop.py
|   |-- dataset
|   |   |-- active_dataset.py
|   |   `-- base.py
|   `-- heuristics
|       |-- heuristics.py
|       `-- utils.py
|-- array_utils.py
|-- dev_configs
|   |-- dataset
|   |   `-- cityscapes_pixel.py
|   |-- fpn_config.py
|-- dist_train_al.sh
|-- hooks
|   |-- __init__.py
|   `-- wandb.py
|-- metrics.py
|-- mmseg_custom
|   |-- __init__.py
|   `-- train.py
|-- model_wrapper.py
|-- runner
|   |-- __init__.py
|   `-- active_learning_runner.py
`-- train_active_learning.py
```