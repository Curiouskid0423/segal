# :rocket: SegAL 

SegAL is a comprehensive active learning baseline system based on [OpenMMLab](https://github.com/open-mmlab) framework, intending to support ***all*** essential features in active learning experiment, which includes providing image-based and pixel-based sampling pipelines, different heuristic functions (e.g. margin sampling, entropy), and visualization features such as "visualize selected pixels". SegAL can be easily integrated with user-defined sampling methods by modifying one of the key files listed below, as well as the up-to-date models from OpenMMLab, provided by [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and more. 

#### Getting started
- Create and edit a config file the way you interact with MMLab modules. Check out tutorials [here](https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html). Then add active learning specific configs following this [guide](./al_config_guide.md), or any files in `experiments/` if you prefer following through examples. The only differences from conventional usages of OpenMMLab are the `active_learning` and `runner` configs. More detailed spec to come.

- Train with command: `bash dist_train.sh <config_file> <number_of_GPUs>`


#### Main functionalities

| Setting | Options |
| --- | --- |
| `benchmark` | **ADA** / **Source-free ADA** |
| `sample schedule` | **regularly** / **irregularly** |
| `sample setting`  | **pixel** / **image** / **region** |
| `sample heuristics`| **entropy** / **margin** / **random** / **ripu** |
| `runner`| **ActiveLearningRunner** / **MultiTaskActiveRunner** |

For detailed examples of how to use the SegAL functionalities, check `experiments/`.

#### Key files
```
almma/              // files of our new method (multitask active learning)
configs/            // mmseg config files
experiments/        // experiment related files
mmseg/              // mmsegmentation source code (version 0.30.0)
dist_train.sh       // training script
dist_train_multiple.sh    // helper script to pipeline training processes
segal/
|-- dataset_wrapper.py
|-- train_active_learning.py
|-- model_wrapper.py
|-- active
|   |-- active_loop.py
|   |-- dataset
|   |   |-- active_dataset.py
|   |   `-- base.py
|   `-- heuristics
|       `-- heuristics.py
|-- method
|   |-- segmentor.py
|   `-- mae_decoder.py
|-- transforms
|   |-- load_mask.py
|   |-- random_crop.py
|   |-- random_flip.py
|   `-- resize.py
|-- losses
|   `-- reconstruction_loss.py
|-- hooks
|   `-- wandb.py
|-- runner
|   |-- active_learning_runner.py
|   `-- multitask_active_runner.py
`-- utils
    `-- train.py
```

#### Development Notes

- **2023.02.27** Added several functionalities:
    - allowed `workflow` to define multiple nested tuples to **sample irregularly**, as did in RIPU paper
    - allowed learning rate to schedule by both `iter` and `epochs` and tested both cases under both `reset_each_round=[True, False]`
    - moved `RIPU_Net` and `Sparsity_Net` onto GPU
    - fixed a critical bug for `ignore_index` in `train_active_learning.py`
    - re-enabled image-based sampling with all the new configurations -- an intermediate step to building *Continuous Active Learning* framework
    - refactored code in `active_learning_runner.py`
    - in process of optimizing GPU memory usage. currently with just 1 sample_per_gpu and DeepLabv3+ R101 backbone could lead to `CUDA_OUT_OF_MEMORY` error constantly at the end of the first epoch. removed uncessary `pool` variable creation in `pixel` sampling. removed unnecessary `deepcopy`.
- **2022.11.11** Provably solved the masking indeterministic behavior by writing a common pickle file in a temporary folder `queries_save_dir`. With the same set of hyperparameters (did not tune much), the model achieves 61.2 mean IoU with 50 epochs on Cityscapes. Full reproduction run of Pixel detailed below.
    | # labelled pixels per img (% annotation)  | mean IoU |
    | ------------- | ------------- |
    | 20 (0.015)  | 56.01  |
    | 40 (0.031)  | 58.33  |
    | 60 (0.046)  | 59.2  |
    | 80 (0.061)  | 61.18  |
    | 100 (0.076) | **61.2**  |

- **2022.10.20** "Rescale" pipeline implemented successfully, and was able to reproduce (approximately) PixelPick result with a different set of hyperparameter (mean IoU of 61). Added feature for "visualization of selected pixels". 
- **2022.09.02** Resolved multiple bugs: (1) weights re-init at each round (2) lr scheduler restart at each round (3) test pipeline applied during sampling instead of train pipeline. 