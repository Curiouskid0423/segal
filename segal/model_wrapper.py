"""
ModelWrapper to contain an instance attribute of 
type torch.nn.Module that MMSeg can work on. 
"""
from collections.abc import Sequence
from typing import Callable, Optional
from functools import reduce

import numpy as np
import pickle
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import mmcv
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info
from mmseg.utils import get_root_logger
from mmseg.datasets import build_dataloader
from segal.active.dataset.active_dataset import ActiveLearningDataset
from segal.active.heuristics import AbstractHeuristic
# from torch.utils.data.dataloader import default_collate
# from mmcv.engine import collect_results_cpu


def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [fn(v) for v in val]
    elif isinstance(val, dict):
        return {k: fn(v) for k, v in val.items()}
    return fn(val)


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result is not None:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


class ModelWrapper:
    """
    Wrapper created to ease the training/testing/loading.

    Args:
        model (nn.Module): The model to optimize.
        criterion (Callable): A loss function.
    """

    def __init__(self, model, cfgs):
        self.backbone = model
        self.logger = get_root_logger()
        self.cfg = cfgs
        self.sample_mode = cfgs.runner.sample_mode
        self.sample_settings = getattr(cfgs.active_learning.settings, self.sample_mode)
        self.gpu_ids = cfgs.gpu_ids
        self.seed = cfgs.seed

    def predict_on_dataset_generator(
        self, dataset, batch_size, iterations, use_cuda, workers=4,
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
        raise NotImplementedError

    def batch_preprocess(self, data_batch):
        """ Preprocess data batch for type compatibility """

        if isinstance(data_batch['img'], DataContainer):
            ext_img = data_batch['img'].data[0].cuda()
        else:
            ext_img = data_batch['img'][0].data[0].cuda()

        if isinstance(data_batch['img_metas'], DataContainer):
            ext_img_meta = data_batch['img_metas'].data[0]
        else:
            ext_img_meta = data_batch['img_metas'][0]

        return ext_img, ext_img_meta

    def predict_on_dataset(
            self, dataset: Dataset, heuristic: AbstractHeuristic, tmpdir="./tmpdir/", **kwargs):
        """
        Make predictions on the unlabelled pool during the sampling phase.
        Image sampling mode:
            returns one value per image, the size of dataset shrinks 
            over time as more samples got labelled.
        Pixel sampling mode:
            returns a h*w boolean map per image, representing new pixels to be labelled.
        """
        self.eval()
        model = self.backbone
        if self.sample_mode == 'pixel':
            assert isinstance(dataset, ActiveLearningDataset)
        self.query_dataset_size = len(dataset)
        test_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            shuffle=False,
            num_gpus=len(self.gpu_ids),
            dist=True if len(self.gpu_ids) > 1 else False,
            seed=self.seed,
            drop_last=False
        )

        results = []

        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))

        if self.sample_mode == 'pixel':
            if hasattr(self.sample_settings, 'sample_evenly'):
                sample_evenly = self.sample_settings.sample_evenly
            else:
                sample_evenly = False

        for idx, data_batch in enumerate(test_loader):

            if self.sample_mode == 'pixel':
                data_batch, mask = data_batch

            with torch.no_grad():
                ext_img, ext_img_meta = self.batch_preprocess(data_batch)
                outputs = model.module.encode_decode(ext_img, ext_img_meta)
                scores = heuristic.get_uncertainties(
                    outputs, 
                    mask = None if self.sample_mode == 'image' else mask
                )
                
                # Cannot store the entire pixel-level map due to memory shortage.
                if self.sample_mode == 'pixel':
                    for i, score in enumerate(scores):
                        score[mask[i].numpy()] = -1.0  # Mask labeled pixels (0 is the lowest uncertainty value)
                        if sample_evenly:
                            new_query = self.extract_query_indices(uc_map=score)
                            results.append(new_query)
                        else:
                            results.append(score)
                elif self.sample_mode == 'image':
                    results.extend(scores)
                else:
                    raise NotImplementedError("sample_mode has to be either pixel or image currently")

            # Rank 0 worker will collect progress from all workers.
            if rank == 0:
                completed = outputs.size()[0] * world_size
                for _ in range(completed):
                    prog_bar.update()

        if hasattr(self.sample_settings, 'sample_evenly') and not sample_evenly:
            results = self.extract_query_indices(np.array(results), sample_evenly=False)

        # collect results from all devices (GPU)
        results = np.array(results, dtype=bool)
        all_results = collect_results_gpu(results, size=np.prod(results.shape)) or []
        all_results = np.array(all_results)

        # For rank 0 worker
        if len(all_results) > 0:
            return all_results

        # For workers that are not rank 0
        if self.sample_mode == 'pixel':
            ds = dataset[0][0]['gt_semantic_seg'][0].data.squeeze().size()
            return np.zeros(shape=(len(dataset), ds[0], ds[1]))
        else:
            return np.zeros(len(dataset))

    def get_pixels_by_budget(self, budget: int, uc_map: torch.Tensor, sample_evenly: bool):
        """
        The method is for pixel-based sampling (not image-based).

        Given uncertainty map or a list of uncertainty maps, budget, and sample_evenly,
        return the top K pixels to label according to the budget and `sample_threshold`
        Returns (value, index) using torch.Tensor.topk API.
        """

        top_k_percent = None
        unit = reduce(lambda x, y: x*y, uc_map.shape)

        if hasattr(self.sample_settings, 'sample_threshold'):
            top_k_percent = self.sample_settings['sample_threshold']
            assert top_k_percent > 0., \
                f"Can only sample with sample_threshold > 0, but received sample_threshold {top_k_percent}"
            selection_pool = int(top_k_percent / 100 * unit)
        else:
            if sample_evenly:
                selection_pool = budget // self.query_dataset_size
            else:
                selection_pool = budget

        if self.cfg.active_learning.heuristic == 'entropy' and hasattr(self.sample_settings, 'entropy_prop'):
            entropy_prop = float(self.sample_settings['entropy_prop'])
            self.logger.info("Using entropy sampling with proportion: ", entropy_prop)
            entropy_selection_pool = int(selection_pool * entropy_prop)
            random_selection_pool = selection_pool - entropy_selection_pool
            values, entropy_indices = uc_map.flatten().topk(k=entropy_selection_pool, dim=-1, largest=True)
            sorted_indices = np.argsort(uc_map.flatten().cpu().numpy())[::-1]
            sorted_indices = sorted_indices[entropy_selection_pool:]
            random_indices = np.random.choice(sorted_indices, size=random_selection_pool, replace=False)
            indices = np.concatenate([entropy_indices.cpu().numpy(), random_indices])
        else:
            values, indices = uc_map.flatten().topk(k=selection_pool, dim=-1, largest=True)
            indices = indices.cpu().numpy()

        if top_k_percent != None:  # sample_threshold=True
            if sample_evenly:
                query_size = budget // unit
                indices = np.random.choice(indices, size=query_size, replace=False)
            else:
                assert len(indices) >= budget
                indices = np.random.choice(indices, size=budget, replace=False)
        indices_of_original_shape = np.unravel_index(indices, uc_map.shape)
        return indices_of_original_shape

    def extract_query_indices(self, uc_map, sample_evenly=True):
        """
        Given either an uncertainty map (e.g. 512 x 1024 for Cityscapes) or a list
        of uncertainty maps, return a set of query indices in corresponding dimensions.
        This methods assumes that the already labeled pixels have been set to the 
        lowest possible uncertainty score upon input (to avoid re-labeling).

        Args:
            uc_map:         A List of map (N, H, W) or an uncertainty score map (H, W)
            sample_evenly:  Indicate whether to sample pixels for labeling evenly across
                            all images. By default to True when uc_map is NOT a list, 
                            i.e. 2 dimension.
        Return:
            query_indices:  A boolean mask to indicate which pixels to label
        """

        assert (sample_evenly and len(uc_map.shape) == 2) or \
            (not sample_evenly and len(uc_map.shape) == 3)

        sample_unit = 'pixel'

        budget = self.sample_settings.budget_per_round
        uc_map_cuda = torch.FloatTensor(uc_map).cuda() 
        if sample_unit == 'pixel':
            indices = self.get_pixels_by_budget(budget, uc_map_cuda, sample_evenly)
        elif sample_unit == 'region':
            indices = self.get_regions_by_budget(budget, uc_map_cuda, sample_evenly)
        else:
            raise NotImplementedError("Not recognized sampling unit.")
        new_query = np.zeros(uc_map.shape, dtype=bool)
        new_query[indices] = True
        return new_query

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
