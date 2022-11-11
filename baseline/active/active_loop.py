"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import numpy as np
from copy import deepcopy
import os
import os.path as osp
import pickle
import torch.utils.data as torchdata
from mmseg.utils import get_root_logger
from mmcv.runner import get_dist_info
from .heuristics import AbstractHeuristic, Random
from .dataset import ActiveLearningDataset
from PIL import Image

CWD = os.getcwd()

class ActiveLearningLoop:

    """
    A loop to synchronously update with the training loop (by unit of epoch)
    Each step will label some datapoints from the unlabeled pool, and add it 
    to the training set (labeled).

    Args:
        dataset:                Type ActiveLearningDataset. Dataset with some sample already labelled.
        query_dataset:          Dataset with query_pipeline applied (same size as mask)
        get_probabilities (Function): 
            Dataset -> **kwargs -> ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic):  Heuristic from baal.active.heuristics.
        query_size (int):       Number of sample to label per step.
        max_sample (int):       Limit the number of sample used (-1 is no limit).
        **kwargs:               Parameters forwarded to `get_probabilities`.
    """

    def __init__(
        self,
        dataset: ActiveLearningDataset,
        query_dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: AbstractHeuristic = Random(),
        configs: dict = {},
        max_sample=-1,
        **kwargs,
    ):

        self.dataset = dataset 
        self.query_dataset = query_dataset
        self.get_probabilities = get_probabilities
        self.heuristic = heuristic
        self.max_sample = max_sample
        self.configs = configs 
        self.sample_mode = configs.runner.sample_mode
        self.has_stepped = False # a variable to control whether to create a file for query_mask saving
        assert self.sample_mode in ['pixel', 'image'], "Sample mode needs to be either pixel or image"
        self.sample_settings = getattr(configs.active_learning.settings, self.sample_mode)
        if self.sample_mode == 'pixel':
            self.num_labelled_pixels =  self.sample_settings['initial_label_pixels']
        self.logger = get_root_logger()

        if hasattr(configs.active_learning, 'visualize'):
            assert configs.active_learning.visualize.size > 0
            assert hasattr(configs.active_learning.visualize, 'dir')
            self.vis_settings = configs.active_learning.visualize
            vis_length = configs.active_learning.visualize.size
            self.vis_path = osp.join(CWD, configs.active_learning.visualize.dir)
            os.makedirs(self.vis_path, exist_ok=True)
            self.vis_indices = [np.random.randint(0, len(self.dataset)) for _ in range(vis_length)]
            self.round = 0

            self.visualize()
        
        self.kwargs = kwargs

    def update_image_labelled_pool(self, dataset, indices):
        
        if len(dataset) <= 0: 
            return False

        uncertainty_scores = self.get_probabilities(dataset, self.heuristic, **self.kwargs)

        if uncertainty_scores is not None:
            ranked = self.heuristic.reorder_indices(uncertainty_scores)
            if indices is not None:
                ranked = indices[np.array(ranked)]
            if len(ranked) > 0:
                self.dataset.label(ranked[:self.sample_settings.query_size])
                return True

        return False

    def update_pixel_labelled_pool(self):

        rank, world = get_dist_info()

        # single gpu / multi gpu, first pass / multi gpu, late passes
        tmp_dir = osp.join(CWD, 'queries_save_dir')
        file_name = f'query_rank_0.pkl'
        if not self.has_stepped:
            # skip the first step() call since no file to read from
            self.has_stepped = True
        elif world > 1 and rank != 0: # multi-gpu setting
            # if not rank=0 worker, load the new query mask from `tmp_dir`
            file_path = osp.join(tmp_dir, file_name)
            assert osp.exists(file_path), f'query mask file save path not exist ({file_path})'
            with open(file_path, 'rb') as fs:
                updated_mask = pickle.load(fs)
                assert isinstance(updated_mask, np.ndarray), 'query mask should be saved as np.array type.'
                print(f" [rank {rank}] loaded updated_mask, shape ", updated_mask.shape)
                print(f" [rank {rank}] updated_mask nonzero() count ", updated_mask.nonzero()[0].shape)
                self.query_dataset.masks = updated_mask

        query_pixel_map = self.get_probabilities(self.query_dataset, self.heuristic, **self.kwargs)
        query_pixel_map = query_pixel_map[:len(self.dataset.masks)] 

        # Set train_set's mask to be the masks computed on query_set
        new_query_mask = np.logical_or(self.query_dataset.masks, query_pixel_map)

        if rank == 0:
            self.dataset.masks = new_query_mask
            self.query_dataset.masks = deepcopy(new_query_mask)
            # save the query file for other devices to collect from
            if world > 1:
                os.makedirs(tmp_dir, exist_ok=True)
                with open(osp.join(tmp_dir, file_name), 'wb') as fs:
                    pickle.dump(new_query_mask, fs)
                
        self.num_labelled_pixels += self.sample_settings['query_size']
        
        return True

    def step(self, pool=None) -> bool:
        """
        Sample and annotate from the pool at each step
        
        Return: 
        True if successfully stepped, False if not (thus stop traing)
        """
        
        if self.sample_mode == 'image':
            pool = self.dataset.pool
            assert pool is not None, "self.dataset.pool should not be None"

            if len(pool) > 0:
                # Whether max sample size is capped
                if self.max_sample != -1 and self.max_sample < len(pool):
                    # Sample without replacement
                    indices = np.random.choice(len(pool), self.max_sample, replace=False)
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
                return self.update_image_labelled_pool(dataset=pool, indices=indices)

        elif self.sample_mode == 'pixel':
            return self.update_pixel_labelled_pool()

        else:
            raise ValueError(f"Sample mode {self.sample_mode} is not supported.")

        return False

    def vis_in_comparison(self, file_name: str, ori: np.ndarray, mask: np.ndarray):
        h, w = mask.shape
        result = Image.new('RGB', (w*2, h))
        ori = Image.fromarray(ori).resize((w, h))
        result.paste(im=ori, box=(0,0))
        result.paste(im=Image.fromarray(mask), box=(w, 0))
        result.save(file_name)

    def vis_in_overlay(self, file_name: str, ori: np.ndarray, mask: np.ndarray):
        h, w = mask.shape
        ori = Image.fromarray(ori, 'RGB').resize((w, h))
        vis_mask = mask.astype(np.uint8)[..., None].repeat(3, axis=-1)
        vis_mask[vis_mask == 1] = 255 # (256, 512, 3) RGB image
        vis_mask = Image.fromarray(vis_mask, 'RGB')
        result = Image.blend(ori, vis_mask, alpha=0.5)
        result.save(file_name)
        
    def revert_transforms(self, original_image, img_metas):
        # normalization configs
        norm_cfg = img_metas['img_norm_cfg']
        mean, std = norm_cfg['mean'][None, None, :], norm_cfg['std'][None, None, :]
        ori = original_image['img'].data.permute(1,2,0).cpu()
        # Un-normalize original image tensor and convert to NumPy
        if ('flip' in img_metas) and img_metas['flip']:
            axis = [1] if (img_metas['flip_direction'] == 'horizontal') else [0]
            ori = ori.flip(dims=axis) # flip the image back for display
        ori = (ori.numpy() * std + mean).astype(np.uint8)
        return ori
                
    def visualize(self):
        """
        Visualize the selected pixels on randomly selected images.
        """

        rank, _ = get_dist_info()

        if rank == 0:
            epoch_vis_dir = osp.join(self.vis_path, f"round{self.round}")
            os.makedirs(epoch_vis_dir, exist_ok=True)
            self.logger.info("Saving visualization...")
            for v in self.vis_indices:
                # original image (256x512 for Cityscapes) and corresponding mask
                ori, mask = self.dataset.get_raw(v), self.dataset.masks[v]
                ori = self.revert_transforms(ori, ori['img_metas'].data)
                file_name = osp.join(epoch_vis_dir, f"{v}.png")
                
                if hasattr(self.vis_settings, "overlay") and self.vis_settings.overlay:
                    self.vis_in_overlay(file_name, ori, mask)
                else:
                    self.vis_in_comparison(file_name, ori, mask)

            self.round += 1