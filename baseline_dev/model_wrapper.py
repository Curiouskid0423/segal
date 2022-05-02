"""
ModelWrapper to contain an instance attribute of 
type torch.nn.Module that MMSeg can work on. 
"""
import sys
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import torch
import mmcv
from mmcv.runner import get_dist_info
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmseg.datasets import build_dataloader
# from torch.utils.data.dataloader import default_collate

def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [fn(v) for v in val]
    elif isinstance(val, dict):
        return {k: fn(v) for k, v in val.items()}
    return fn(val)


class ModelWrapper:
    """
    Wrapper created to ease the training/testing/loading.

    Args:
        model (nn.Module): The model to optimize.
        criterion (Callable): A loss function.
    """

    def __init__(self, model, logger=None, cfg=None):
        self.backbone = model
        assert logger is not None, "logger argument should be provided"
        self.logger = logger    
        self.cfg = cfg
        
    def predict_on_dataset_generator(
        self, dataset, batch_size, iterations, use_cuda, workers = 4,
        collate_fn: Optional[Callable] = None,
        half=False, verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.

        Args:
            dataset:    Dataset to predict on.
            batch_size: Batch size to use during prediction.
            iterations: Number of iterations per sample.
            use_cuda:   Use CUDA or not.
            workers:    Number of workers to use.
            collate_fn: The collate function to use.
            half:       If True use half precision.
            verbose:    If True use tqdm to display progress

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        self.eval()
        pass

    def predict_on_dataset(self, dataset, heuristic, tmpdir="./tmpdir/", **kwargs):
        
        # self.eval()
        # model = self.backbone

        # test_loader = build_dataloader(
        #     dataset,
        #     self.cfg['data'].samples_per_gpu,
        #     self.cfg['data'].workers_per_gpu,
        #     shuffle=False,
        #     num_gpus= 1, #len(self.cfg['gpu_ids']),
        #     dist=False, #True if len(self.cfg['gpu_ids']) > 1 else False,
        #     seed=self.cfg['seed'],
        #     drop_last=True,
        #     )

        # results = torch.Tensor().cpu()

        # rank, world_size = get_dist_info()
        # if rank == 0:
        #     prog_bar = mmcv.ProgressBar(len(dataset))

        # for batch in test_loader:
        #     with torch.no_grad():

        #         batch.pop('gt_semantic_seg') # delete the ground truth from batch
        #         ext_img = batch['img'].data[0].cpu() #.cuda()
        #         ext_img_meta = batch['img_metas'].data[0]
                
        #         # NOTE Approach 1: This gives pixel-classified result via `cls_seg` call;
        #         # outputs = model(return_loss=False, rescale=True, **batch)
        #         # NOTE Approach 2:
        #         # outputs dim: Size([2, 19, 512, 1024]) # 2 is batch_size; 19 classes;
        #         outputs = model.module.encode_decode(ext_img, ext_img_meta) #.cpu()
                
        #         scores = heuristic.get_uncertainties(outputs).cpu()
        #         results = torch.cat((results, scores), dim=0)
                
        #     # rank 0 worker will collect progress from all workers.
        #     if rank == 0:
        #         completed = outputs.size()[0] * world_size
        #         for _ in range(completed):
        #             prog_bar.update()
            
        # return results

        self.eval()
        model =self.backbone
        
        test_loader = build_dataloader(
            dataset,
            self.cfg['data'].samples_per_gpu,
            self.cfg['data'].workers_per_gpu,
            shuffle=False,
            num_gpus= len(self.cfg['gpu_ids']),
            dist=True if len(self.cfg['gpu_ids']) > 1 else False,
            seed=self.cfg['seed'],
            drop_last=True,
            )

        # results = torch.Tensor()
        results = []

        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))

        for batch in test_loader:
            with torch.no_grad():
                batch.pop('gt_semantic_seg') # delete the ground truth from batch
                ext_img = batch['img'].data[0].cuda()
                ext_img_meta = batch['img_metas'].data[0]
                outputs = model.module.encode_decode(ext_img, ext_img_meta)
                scores = heuristic.get_uncertainties(outputs).cpu().numpy()
                # scores = [0] # DEBUG
                # results = torch.cat((results, scores), dim=0)
                results.extend(scores)
                
            # rank 0 worker will collect progress from all workers.
            if rank == 0:
                completed = outputs.size()[0] * world_size
                for _ in range(completed):
                    prog_bar.update()

        # collect results from all devices (GPU) to cpu directory
        all_results = collect_results_gpu(results, size=len(dataset))
        if all_results is not None:
            return np.array(all_results)
            

    def get_params(self):
        """
        Return the parameters to optimize.
        """
        return self.backbone.parameters()

    def state_dict(self):
        """Get the state dict(s)."""
        return self.backbone.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load the model with `state_dict`."""
        self.backbone.load_state_dict(state_dict, strict=strict)

    def train(self):
        """Set the model in `train` mode."""
        self.backbone.train()

    def eval(self):
        """Set the model in `eval mode`."""
        self.backbone.eval()