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
            # img_dir='images',
            img_suffix='.png',
            # ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)