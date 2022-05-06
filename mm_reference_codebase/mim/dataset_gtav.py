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


@DATASETS.register_module()
class GTAVEPEDataset(CustomDataset):
    """GTAV EPE dataset.
    https://github.com/intel-isl/PhotorealismEnhancement

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_cityscapes.png' for GTAV EPE dataset.
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
        super(GTAVEPEDataset, self).__init__(
            img_dir='images_epe',
            img_suffix='.jpg',
            ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            split='epe.txt',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)


@DATASETS.register_module()
class GTAVCYCLEDataset(CustomDataset):
    """GTAV CYCLE dataset.
    https://github.com/jhoffman/cycada_release

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_cityscapes.png' for GTAV CYCLE dataset.
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
        super(GTAVCYCLEDataset, self).__init__(
            img_dir='images_cycle',
            img_suffix='.jpg',
            ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            split='cycle.txt',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)


@DATASETS.register_module()
class GTAVMUNITDataset(CustomDataset):
    """GTAV MUNIT dataset.
    https://github.com/NVlabs/MUNIT

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_cityscapes.png' for GTAV MUNIT dataset.
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
        super(GTAVMUNITDataset, self).__init__(
            img_dir='images_munit',
            img_suffix='.jpg',
            ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            split='munit.txt',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)


@DATASETS.register_module()
class GTAVCUTDataset(CustomDataset):
    """GTAV CUT dataset.
    http://taesung.me/ContrastiveUnpairedTranslation/

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_cityscapes.png' for GTAV CUT dataset.
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
        super(GTAVCUTDataset, self).__init__(
            img_dir='images_cut',
            img_suffix='.jpg',
            ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            split='cut.txt',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)


@DATASETS.register_module()
class GTAVCTDataset(CustomDataset):
    """GTAV ColourTransfer dataset.
    http://erikreinhard.com/colour_transfer.html

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_cityscapes.png' for GTAV ColourTransfer dataset.
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
        super(GTAVCTDataset, self).__init__(
            img_dir='images_ct',
            img_suffix='.jpg',
            ann_dir='labels',
            seg_map_suffix='_cityscapes.png',
            split='ct.txt',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)
