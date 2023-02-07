# :rocket: SegAL 

SegAL is a comprehensive active learning baseline system based on [OpenMMLab](https://github.com/open-mmlab) framework, intending to support ***all*** essential features in active learning experiment, which includes providing image-based and pixel-based sampling pipelines, different heuristic functions (e.g. margin sampling, entropy), and visualization features such as "visualize selected pixels". SegAL can be easily integrated with user-defined sampling methods by modifying one of the key files listed below, as well as the up-to-date models from OpenMMLab, provided by [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and more. 

#### Getting started
- Create and edit a config file the way you interact with MMLab modules. Check out tutorials [here](https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html). Then add active learning specific configs following this [guide](./al_config_guide.md), or any files in `experiments/` if you prefer following through examples. The only differences from conventional usages of OpenMMLab are the `active_learning` and `runner` configs. More detailed spec to come.

- Train with command: `bash dist_train.sh <config_file> <number_of_GPUs>`

#### Key files
```
configs/            // mmseg config files
experiments/        // experiment related files
dist_train.sh    // training script
segal/
|-- train_active_learning.py
|-- model_wrapper.py
|-- active
|   |-- active_loop.py
|   |-- dataset
|   |   |-- active_dataset.py
|   |   `-- base.py
|   `-- heuristics
|       |-- heuristics.py
|       `-- utils.py
|-- hooks
|   `-- wandb.py
|-- utils
|   `-- train.py
`-- runner
    `-- active_learning_runner.py
```
#### Development Notes
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


#### Action Items
- Implement `RandomCrop` augmentation.
- Integrate `MMSelfSup` to allow SSL features in active learning query. 
- Refactor as needed to accommodate the case where we gradually increase *BOTH labeled pixels and images.*
