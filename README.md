# :rocket: SegAL 

SegAL is a comprehensive active learning baseline system based on [OpenMMLab](https://github.com/open-mmlab) framework, intending to support ***all*** essential features in active learning experiment, which includes providing image-based and pixel-based sampling pipelines, different heuristic functions (e.g. margin sampling, entropy) and easy integration with new models, given the abundant models supplied in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and more. 

#### Getting started
- Create and edit a config file (the way you interact with MMLab modules; check out tutorials [here](https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html)). Follow the example in `dev_configs/fpn-r50.py`. The only differences from conventional usages of OpenMMLab are the `active_learning` and `runner` configs. More detailed spec to come.

- Train with command: `bash baseline_dev/dist_train_al.sh <config_file> <number_of_GPUs>`

#### Key files
```
baseline/
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
#### Development Notes
- **2022.10.20** "Rescale" pipeline implemented successfully, and was able to reproduce (approximately) PixelPick result with a different set of hyperparameter (**mean IoU of 61**).
- **2022.09.02** Resolved multiple bugs: (1) weights re-init at each round (2) lr scheduler restart at each round (3) test pipeline applied during sampling instead of train pipeline. 


#### Action Items
- Enable visualization of the sampled pixels at each round on a segmentation map.
- Implement `RandomCrop` augmentation.
- Integrate `MMSelfSup` to allow SSL features in active learning query. 
- Add documentation to essential functions and "how to use config file".
- Refactor as needed to accommodate the case where we gradually increase *BOTH labeled pixels and images.*
