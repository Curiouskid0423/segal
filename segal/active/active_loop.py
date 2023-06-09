"""
Should contain ActiveLearningLoop compatible to MMSeg. This should work like 
any step() methods such as StepLR or optimizer()
Code mostly similar to BAAL.
"""
from typing import Callable
import numpy as np
from datetime import datetime
import os
import os.path as osp
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as torchdata
import matplotlib.pyplot as plt
import mmcv
from mmcv.runner import master_only
from mmseg.utils import get_root_logger
from mmseg.apis.inference import show_result_pyplot                    
from .heuristics import AbstractHeuristic, Random, MaskPredictionScore
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
        result = Image.blend(ori, vis_mask, alpha=0.6)
        result.save(file_name)
        
    def onehot_with_ignore_label(self, labels, num_classes, ignore_label):
        dummy_label = num_classes + 1
        mask = (labels == ignore_label) # mask==1 should be removed
        modified_labels = labels.clone()
        modified_labels[mask] = num_classes
        # One-hot encode the modified labels
        one_hot_labels: torch.Tensor = F.one_hot(modified_labels, num_classes=dummy_label)
        # Remove the last row in the one-hot encoding
        one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)
        one_hot_labels = one_hot_labels[:, :-1, :, :]
        return one_hot_labels.to(float)

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
    
    def get_loss_map(self, img, gt, logit, file_name, ignore_label=255,
                     cmap1='gray', cmap2='viridis', alpha=0.9):
        """ Plot the loss pixel-map given a logit (C, H, W) and a ground_truth
        shape [C, H, W]
        """
        if len(logit.shape) == 4:
            logit = logit.squeeze(0) # (C=19, H, W)  (19, 640, 1138)

        # print(f"[get_loss_map] img shape: {img.shape}")
        # print(f"[get_loss_map] gt shape: {gt.shape}")
        # print(f"[get_loss_map] logit shape: {logit.shape}")
        
        one_hot_gt = self.onehot_with_ignore_label(gt, num_classes=len(logit), ignore_label=ignore_label)
        one_hot_gt = one_hot_gt.squeeze(0)
        # print(f"one_hot_gt shape: {one_hot_gt.shape}")
        softmax_pred = F.softmax(logit, dim=0)
        log_loss = -(one_hot_gt * softmax_pred.log()).mean(dim=0, keepdim=False).cpu() # (H, W)

        # create error plots (overlay log_loss on top of the image)
        plt.figure(figsize=(15, 10))
        plt.title(f'Cross-entropy Loss Map ({file_name.split("/")[-1]})')
        plt.imshow(img, cmap=cmap1)
        plt.imshow(log_loss, cmap=cmap2, alpha=alpha)
        plt.colorbar()
        plt.savefig(file_name)
        plt.close()
        
    def get_error_map(self, img, logit: torch.Tensor, gt: torch.Tensor, file_name, 
                      ignore_label=255, cmap1='gray', cmap2='viridis', alpha = 0.7):
        
        if len(logit.shape) == 4:
            logit = logit.squeeze(0) # (C=19, H, W)  (19, 640, 1138)
        
        pred = F.softmax(logit, dim=0)
        pred = pred.argmax(dim=0, keepdim=True) # (1, H, W), value from 0 ~ 18
        gt = gt.squeeze(0) # (1, H, W)
        # print(f"pred shape: {pred.shape}")
        # print(f"gt shape: {gt.shape}")
    
        err_mask = (gt != pred) & (gt != ignore_label) # True means "is an error"

        # print(f"err_mask shape: {err_mask.shape}") # (1, 640, 1138)
        err_mask = err_mask.permute(1,2,0).to(int).cpu()
        err_mask[err_mask == 1] = 255 # change color value to white

        # create error plots (overlay log_loss on top of the image)
        plt.figure(figsize=(15, 10))
        plt.title(f'Misprediction / Error Map ({file_name.split("/")[-1]})')
        plt.imshow(img, cmap=cmap1)
        plt.imshow(err_mask, cmap=cmap2, alpha=alpha)
        plt.colorbar()
        plt.savefig(file_name)
        plt.close()
        
    @master_only
    def vis_uncertainty(
        self, img, scores, file_name, alpha=0.5, 
        cmap1='gray', cmap2='viridis', overlay_img=True):
        
            
        # type conversions
        if isinstance(scores, torch.Tensor):
            scores = scores.permute(1,2,0).cpu().numpy()
        if len(scores.shape) == 3:
            scores = scores.squeeze(0)

        # create uncertainty plots 
        plt.figure(figsize=(15, 10))
        plt.title(f'Uncertainty Map ({file_name.split("/")[-1]})')
        if overlay_img:
            plt.imshow(img, cmap=cmap1)
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
            
            if model != None:
                # model.eval() should be called in runner file already
                with torch.no_grad():
                    ext_img, ext_img_meta = ori['img'].data, ori['img_metas'].data
                    logits = model.module.whole_inference(
                        ext_img.unsqueeze(0).cuda(), ext_img_meta, rescale=False) # (B, C, H, W)
                    if isinstance(self.heuristic, MaskPredictionScore):
                        scores = self.heuristic.compute_score(
                            network=model.module, image=ext_img.unsqueeze(0).cuda(), seg_logit=logits)
                    else:
                        scores = self.heuristic.get_uncertainties(logits) # 100 bins

                    
                    uncertainty_fname = osp.join(epoch_vis_dir, f"{v}_uncertainty.png")
                    qt_uncertainty_fname = osp.join(epoch_vis_dir, f"{v}_uncertainty_qt.png")
                    loss_map_fname = osp.join(epoch_vis_dir, f"{v}_loss_map.png")
                    error_map_fname = osp.join(epoch_vis_dir, f"{v}_error_map.png")

                    original_image = self.revert_transforms(ext_img, ext_img_meta)

                    # visualize uncertainty score map                    
                    self.vis_uncertainty(original_image, scores, file_name=uncertainty_fname, alpha=0.9)

                    # quantization section
                    bin_size = 0.05 # 20 bins 
                    bins = np.arange(0, 1.0, bin_size)
                    score_quantized = np.digitize(scores, bins, right=True) * bin_size
                    self.vis_uncertainty(
                        original_image, score_quantized, file_name=qt_uncertainty_fname, alpha=0.9)

                    # visualize the cross-entropy loss map
                    self.get_loss_map(
                        img = original_image,
                        gt  = ori['gt_semantic_seg'].data.cuda(), 
                        logit = logits, 
                        file_name = loss_map_fname)

                    # visualize the errors (i.e. mispredictions)
                    self.get_error_map(
                        img=original_image, 
                        logit=logits, 
                        gt=ori['gt_semantic_seg'].data.cuda(), 
                        file_name=error_map_fname)

                    # visualize the segmentation map
                    result = torch.argmax(logits, dim=1, keepdim=False) # (B, H, W)
                    ext_img = ext_img.permute(1,2,0) # (H, W, C)
                    show_result_pyplot(
                        model=model.module, 
                        img=ext_img.cpu().numpy(), 
                        result=result.cpu().numpy(),
                        palette=self.query_dataset.dataset.PALETTE,
                        out_file=osp.join(epoch_vis_dir, f"{v}_segmap.png"))

                    plt.close('all')

        self.round += 1