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
from mmseg.datasets import build_dataloader
from tqdm import tqdm
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

    def predict_on_dataset(self, dataset, **kwargs):
        
        self.eval()
        model = self.backbone
        test_loader = build_dataloader(
            dataset,
            self.cfg['data'].samples_per_gpu,
            self.cfg['data'].workers_per_gpu,
            len(self.cfg['gpu_ids']),
            dist=True if len(self.cfg['gpu_ids']) > 1 else False,
            seed=self.cfg['seed'],
            drop_last=True,
            )

        results = []
        self.logger.info(f"Predicting on unlabeled pool. Pool size {len(dataset)}")
        # FIXME: Use `test_pipeline` config ?
        # FIXME: Some config params in cfg.active_learning is not used.
        # for batch in test_loader:
        #     with torch.no_grad():
        #         batch['img'] = batch['img'].data # .numpy().tolist()
        #         batch['img_metas'] = batch['img_metas'].data
        #         batch.pop('gt_semantic_seg') # delete the ground truth from batch
        #         outputs = model(return_loss=False, **batch)
        #     results.extend(outputs)
        # FIXME: Testing if iterating over test_loader is the source of leaked semaphores.
        results = [0 for _ in range(len(test_loader))]
        return np.array(results)

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