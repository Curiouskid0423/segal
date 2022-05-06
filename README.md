# Active Learning Baseline System 

- **2022.05.02.** Current system works with single GPU training, as well as two heuristics options -- random and mean-entropy. Distributed training mode contains some memory issue within step() function to be fixed.

#### Getting started
- Create and edit a config file (the way you interact with MMLab modules)
- Train with command:`bash baseline_dev/dist_train_al.sh <script>`

#### Key files
```
|-- active
|     |-- active_loop.py
|     |-- dataset
|     |     |-- active_dataset.py
|     |     `-- base.py
|     `-- heuristics
|           |-- heuristics.py
|           `-- utils.py
|-- dist_train_al.sh
|-- fcn_test_config.py
|-- hooks
|     `-- wandb.py
|-- mmseg_custom
|     `-- train.py
|-- model_wrapper.py
|-- runner
|     `-- active_learning_runner.py
`-- train_active_learning.py
```