# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.cityscapes import CityscapesDataset

@DATASETS.register_module()
class ACDCDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCDataset, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
        assert osp.exists(self.img_dir)
        assert osp.exists(self.ann_dir)