# Semantic Segmentation: from gtav to cityscapes

In this repo, we will show how to set up the baseline of semantic segmentation from gtav to cityscapes. There are basically five steps to go as below, and we will do it step by step.

0. Install OpenMMLab libraries
1. Implement a new dataset class
3. Modify config file to use it
4. Train and test a model

The key files are listed as below

```
gtav2cityscales
├── README.md
├── configs
│   ├── gtav2cityscapes
│   │   └── fcn_hr18_512x1024_40k_gtav2cityscapes.py
├── data
│   ├── gtav
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   └── gtFine
├── work_dirs
│   ├── fcn_hr18_512x1024_40k_gtav2cityscapes
│   │   ├── fcn_hr18_512x1024_40k_gtav2cityscapes.log
│   │   ├── fcn_hr18_512x1024_40k_gtav2cityscapes.pth
│   │   └── fcn_hr18_512x1024_40k_gtav2cityscapes.py
├── gtav.py
└── wandb.py
```

## Install OpenMMLab libraries

> **Requirements**
> - Python 3.6+
> - PyTorch 1.10+

```bash
# 中国大陆用户请参考 https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/ 修改 .condarc
conda create -n mim python=3.7 -y && conda activate mim
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -c pytorch

pip install openmim wandb
mim install mmseg -f https://github.com/open-mmlab/mmsegmentation.git
# 中国大陆用户请使用 mim install mmseg -f https://github.com.cnpmjs.org/open-mmlab/mmsegmentation.git
```

## Implement a new dataset class

Then we need to implement a new dataset class `GTAVDataset`, the key implementation of the class is as below.

```python
import os.path as osp
import mmcv
from mmseg.datasets import CustomDataset, DATASETS


@DATASETS.register_module()
class GTAVDataset(CustomDataset):
    """GTAV dataset.
    https://download.visinf.tu-darmstadt.de/data/from_games/index.html

    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is
    fixed to '_cityscapes.png' for GTAV dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):
        super(GTAVDataset, self).__init__(
            img_dir='images',
            img_suffix='.png',
            ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)
```

## Modify config

The key step is to define the `custom_imports` so that MMCV will import the file specified in the list of `imports` when loading the config.
This will load the file `dataset_gtav.py` we implemented in the previous step
so that the `GTAVDataset` can be registered into the `DATASETS` registry in MMDetection.

```python
custom_imports = dict(
    imports=['dataset_gtav'],
    allow_failed_imports=False)
```

The config to run a model can be found in `configs/gtav2cityscapes/fcn_hr18_512x1024_40k_gtav2cityscapes.py` which defined the model and dataset to run.

## Train and test the model

Finally, we can train and evaluate the model through the following command

```bash
PYTHONPATH=$PWD:$PYTHONPATH mim train mmseg configs/gtav2cityscapes/fcn_hr18_512x1024_40k_gtav2cityscapes.py --launcher pytorch --gpus 8 --work-dir work_dirs/fcn_hr18_512x1024_40k_gtav2cityscapes

PYTHONPATH=$PWD:$PYTHONPATH mim test mmseg configs/gtav2cityscapes/fcn_hr18_512x1024_40k_gtav2cityscapes.py --launcher pytorch --gpus 8 --checkpoint work_dirs/fcn_hr18_512x1024_40k_gtav2cityscapes/latest.pth --eval mIoU
```
