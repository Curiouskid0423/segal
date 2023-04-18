"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import os.path as osp
import pickle
import torch
import torch.utils.data as torchdata
import mmcv
from mmcv.runner import master_only
from mmseg.utils import get_root_logger
from .heuristics import AbstractHeuristic, Random
from .dataset import ActiveLearningDataset
from PIL import Image
from segal.utils.misc import newline_after_pbar

CWD = os.getcwd()

class ActiveLearningLoop:

    """
    A loop to synchronously update with the training loop (by unit of epoch)
    Each step will label some datapoints from the unlabeled pool, and add it 
    to the training set (labeled).
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
        assert self.sample_mode in ['pixel', 'region', 'image'], \
            "Sample mode needs to be either pixel or image"
        self.sample_settings = getattr(configs.active_learning.settings, self.sample_mode)
        if self.sample_mode in ['pixel', 'region']:
            self.num_labelled_pixels =  self.sample_settings['initial_label_pixels']
        self.logger = get_root_logger()
        self.time_hash = datetime.now().strftime("%m%d_%H_%M")

        if hasattr(configs.active_learning, 'visualize'):
            assert configs.active_learning.visualize.size > 0
            assert hasattr(configs.active_learning.visualize, 'dir')
            
            self.vis_settings = configs.active_learning.visualize
            vis_length = configs.active_learning.visualize.size

            # create visualization directory within the `work_dir`
            self.vis_path = osp.join(CWD, configs.active_learning.visualize.dir)
            os.makedirs(self.vis_path, exist_ok=True)
            self.vis_indices = [np.random.randint(0, len(self.query_dataset)) for _ in range(vis_length)]
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
                self.dataset.label(ranked[:self.sample_settings.budget_per_round])
                return True

        return False

    def update_pixel_labelled_pool(self):

        query_pixel_map = self.get_probabilities(self.query_dataset, self.heuristic, **self.kwargs)
        query_pixel_map = query_pixel_map[:len(self.query_dataset)] 
        # write the updated masks into their respective paths
        self.update_masks_to_files(query_pixel_map)
        self.num_labelled_pixels += self.sample_settings['budget_per_round']
        
        return True

    def get_mask_fname_by_idx(self, x):
        return self.query_dataset[x]['img_metas'].data['mask_filename']

    @master_only
    def update_masks_to_files(self, new_maps):

        self.logger.info(f"saving updated masks...")
        pbar = mmcv.ProgressBar(task_num=len(new_maps))
        assert len(new_maps) == len(self.query_dataset)
    
        for idx, mask in enumerate(new_maps):
            mask_filename = self.get_mask_fname_by_idx(idx)
            # read the current masks
            with open(mask_filename, 'rb') as fs:
                old_mask = pickle.load(fs)
                new_mask = np.logical_or(old_mask, mask)
            # write the new masks after XOR operation
            with open(mask_filename, 'wb') as fs:
                pickle.dump(new_mask, fs)
            if idx % 900 == 0:
                with open(mask_filename, 'rb') as fs:
                    old_count, new_count = np.count_nonzero(old_mask), np.count_nonzero(pickle.load(fs))
                    print(f" >> [{idx}] current mask: {old_count}, after XOR operator: {new_count}")
                
            pbar.update()
        
        newline_after_pbar(rank=0)

    def step(self, pool=None) -> bool:
        """
        Sample and annotate from the pool at each step
        
        Return: 
        True if successfully stepped, False if not (thus stop training)
        """

        if self.sample_mode == 'image':
            if pool == None:
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

        elif self.sample_mode in ['pixel', 'region']:
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
        ori = original_image.permute(1,2,0).cpu()
        # Un-normalize original image tensor and convert to NumPy
        if ('flip' in img_metas) and img_metas['flip']:
            axis = [1] if (img_metas['flip_direction'] == 'horizontal') else [0]
            ori = ori.flip(dims=axis) # flip the image back for display
        ori = (ori.numpy() * std + mean).astype(np.uint8)
        return ori
    
    @master_only
    def vis_uncertainty(
        self, img, img_metas, scores, file_name, alpha=0.5, 
        cmap1='gray', cmap2='viridis', overlay_img=True):
        
        img_np = self.revert_transforms(img, img_metas)
            
        # type conversions
        if isinstance(scores, torch.Tensor):
            scores = scores.permute(1,2,0).cpu().numpy()
        if len(scores.shape) == 3:
            scores = scores.squeeze(0)

        # create uncertainty plots 
        if overlay_img:
            plt.imshow(img_np, cmap=cmap1)
        plt.imshow(scores, cmap=cmap2, alpha=alpha)
        plt.colorbar()
        plt.savefig(file_name)
        plt.close()

    @master_only  
    def visualize(self, model=None):
        """
        Visualize the selected pixels on randomly selected images.
        """

        # rank, _ = get_dist_info()

        epoch_vis_dir = osp.join(self.vis_path, f"round{self.round}")
        os.makedirs(epoch_vis_dir, exist_ok=True)
        self.logger.info("saving visualization...")
        for v in self.vis_indices:
            # visualize masks
            ori, mask_filename = self.query_dataset.get_raw(v), self.get_mask_fname_by_idx(v)
            with open(mask_filename, 'rb+') as fs:
                mask = pickle.load(fs)
            unnormalized_image = self.revert_transforms(ori['img'].data, ori['img_metas'].data)
            file_name = osp.join(epoch_vis_dir, f"{v}.png")
            if hasattr(self.vis_settings, "overlay") and self.vis_settings.overlay:
                self.vis_in_overlay(file_name, unnormalized_image, mask)
            else:
                self.vis_in_comparison(file_name, unnormalized_image, mask)
            
            # visualize prediction uncertainty map
            if model != None:
                # model.eval() should be called in runner file already
                with torch.no_grad():
                    ext_img, ext_img_meta = ori['img'].data, ori['img_metas'].data
                    logits = model.module.whole_inference(
                        ext_img.unsqueeze(0).cuda(), ext_img_meta, rescale=False)
                    scores = self.heuristic.get_uncertainties(logits)
                    file_name = osp.join(epoch_vis_dir, f"{v}_uncertainty.png")
                    self.vis_uncertainty(ext_img, ext_img_meta, scores, file_name=file_name, alpha=0.7)
    
        self.round += 1