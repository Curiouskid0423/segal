"""
Load Mask
"""
import numpy as np
import torch
import os.path as osp
import pickle
import io
import mmcv
from mmseg.datasets import PIPELINES

@PIPELINES.register_module()
class LoadMasks(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self, mask_dir, file_client_args=dict(backend='disk')):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.mask_dir = mask_dir
        
    def __call__(self, results: dict):
        """Call function to load masks given the dictionary `results`

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        
        get_mask_fname = lambda img_fname: f"{img_fname.split('/')[-1].split('.')[0]}.pkl"

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        assert 'filename' in results.keys()
        raw_mask_fname = get_mask_fname(results['filename'])
        mask_filename = osp.join(self.mask_dir, raw_mask_fname)

        if osp.exists(mask_filename):
            assert osp.getsize(mask_filename) > 0, "found an empty mask file"
            mask_bytes = self.file_client.get(mask_filename)
            with io.BytesIO(mask_bytes) as buff:
                mask_object = pickle.load(buff) # e.g. np.array with shape 512x256
            results['mask'] = torch.from_numpy(mask_object)
            results['mask'].requires_grad_(False) # masks do not need gradients
        else:
            results['mask'] = torch.zeros((100,100)) # placeholder during mask creation
        
        results['mask_filename'] = mask_filename

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mask_dir={self.mask_dir})'
        return repr_str