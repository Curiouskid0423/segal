"""
ModelWrapper to contain an instance attribute of 
type torch.nn.Module that MMSeg can work on. 
"""
from functools import reduce

import numpy as np
import torch
from torch.utils.data import Dataset
import mmcv
from mmcv.runner import get_dist_info
from mmseg.utils import get_root_logger
from mmseg.datasets import build_dataloader
from mmseg.models import EncoderDecoder
from segal.active.dataset.active_dataset import ActiveLearningDataset
from segal.active.heuristics import AbstractHeuristic
from segal.utils.sampling import \
    (collect_results_gpu, batch_preprocess, region_selector)
from segal.utils.misc import newline_after_pbar

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
        self.sample_rounds = cfgs.runner.sample_rounds
        self.sample_mode = cfgs.runner.sample_mode
        self.sample_settings = getattr(cfgs.active_learning.settings, self.sample_mode)
        self.gpu_ids = cfgs.gpu_ids
        self.seed = cfgs.seed
        self.expl_counter = -1 # counter for exploration schedule

    def set_sample_evenly(self):

        if self.sample_mode == 'region':
            assert hasattr(self.sample_settings, 'sample_evenly') and self.sample_settings.sample_evenly
            sample_evenly = True
        elif self.sample_mode == 'pixel':
            if hasattr(self.sample_settings, 'sample_evenly'):
                sample_evenly = self.sample_settings.sample_evenly
            else:
                sample_evenly = False
            
        return sample_evenly

    def predict_on_dataset(
        self, dataset: Dataset, heuristic: AbstractHeuristic, **kwargs):
        """
        Make predictions on the unlabelled pool during the sampling phase.
        Image sampling mode:
            returns one value per image, the size of dataset shrinks 
            over time as more samples got labelled.
        Pixel sampling mode:
            returns a h*w boolean map per image, representing new pixels to be labelled.
        """
        self.eval()
        model: EncoderDecoder = self.backbone.module
        self.query_dataset_size = len(dataset)
        G = len(self.gpu_ids)
        
        assert self.sample_mode == 'image' or isinstance(dataset, ActiveLearningDataset)

        sample_evenly = self.set_sample_evenly()
        workers = self.cfg.data.workers_per_gpu
        results = []
        rank, world_size = get_dist_info()

        test_loader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=workers,
            shuffle=False, num_gpus=G, dist=True if G > 1 else False,
            seed=self.seed, drop_last=False)

        self.logger.info(f"computing uncertainty scores...")
        
        if rank == 0: 
            if self.cfg.active_learning.heuristic == 'mps':
                self.expl_counter += 1
                self.logger.info(f'[linear expl schedule] expl_counter = {self.expl_counter}')
            pbar = mmcv.ProgressBar(len(dataset))
            
        for idx, data_batch in enumerate(test_loader):

            with torch.no_grad():
                
                # get logits
                ext_img, ext_img_meta = batch_preprocess(data_batch)
                logits = model.whole_inference(ext_img, ext_img_meta, rescale=False)
                # get uncertainties
                if self.cfg.active_learning.heuristic == 'mps':
                    assert hasattr(model, 'mae_inference')
                    # set mix_factor according to the exploration schedule
                    if self.cfg.active_learning.heuristic == 'mps':
                        if heuristic.expl_schedule == 'constant':
                            mps_mix_factor = 1.
                        elif heuristic.expl_schedule == 'linear':
                            mps_mix_factor = max(self.sample_rounds-self.expl_counter-1, 0) / (self.sample_rounds-1)
                        else:
                            raise NotImplementedError
                    scores = heuristic.compute_score(model, ext_img, logits, mix_factor=mps_mix_factor)
                else:
                    scores = heuristic.get_uncertainties(logits)
                
                # compute query indices
                if self.sample_mode in ['pixel', 'region']:
                    assert len(scores) == 1, "batch size not one for query dataloader"
                    # mask out labeled pixels to avoid reselection
                    score = scores[0]
                    mask = data_batch['mask'][0].numpy()
                    score[mask] = -float('inf') 
                    if sample_evenly:
                        new_query = self.extract_query_indices(uc_map=score)
                        results.append(new_query)
                    else:
                        results.append(score)
                elif self.sample_mode == 'image':
                    results.extend(scores)
                else:
                    raise NotImplementedError
                
            # Rank 0 worker will collect progress from all workers.
            if rank == 0:
                completed = logits.size()[0] * world_size
                for _ in range(completed):
                    pbar.update()
                    
        newline_after_pbar(rank)

        if hasattr(self.sample_settings, 'sample_evenly') and not sample_evenly:
            results = self.extract_query_indices(np.array(results), sample_evenly=False)

        # collect results from all devices (GPU)
        results = np.array(results, dtype=bool)
        if G > 1:
            all_results = collect_results_gpu(results, size=np.prod(results.shape)) or []
            all_results = np.array(all_results)
        else:
            all_results = results

        print(f"[rank {rank}] debugger message from model_wrapper: results  {len(all_results)}")

        # For rank 0 worker
        if len(all_results) > 0:
            return all_results
        
        # For workers that are not rank 0
        if self.sample_mode in ['pixel', 'region']:
            ds = self.cfg.scale_size
            return np.zeros(shape=(len(dataset), ds[0], ds[1]))
        else:
            return np.zeros(len(dataset))

    def get_pixels(self, selection_pool, uc_map):
        """
        Given an uncertainty map, which is 2D (H, W) or 3D (B, H, W), 
        return the indices of the top uncertainty scores in 1D.
        Args:
            selection_pool (int):   number of pixels to be selected
            uc_map (torch.Tensor):  uncertainty map tensor that can be 2D or 3D
        Return:
            1-dimensional indices of the top uncertainty pixels. Reshaping will happen afterwards.
        """
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
            
        return indices

    def get_regions(self, selection_pool: int, uc_map: torch.Tensor, radius: int = 1):
        """
        Given an uncertainty map, which is 2D (H, W) or 3D (B, H, W), 
        return the indices of the most uncertain regions in a 1-dimension vector

        Args:
            selection_pool (int):   number of pixels to be selected
            uc_map (torch.Tensor):  uncertainty map tensor that can be 2D or 3D
            radius (int):           set the sampling unit as `(2*radius+1)**2`
        Return:
            1-dimensional indices of the top uncertainty pixels. Reshaping will happen afterwards.
        """
        indices = region_selector(selection_pool, uc_map, radius, sample_evenly=True)
        assert len(indices)==1, "`indices` should be 1-dimensional"
        return indices[0]

    def query_by_budget(self, budget: int, uc_map: torch.Tensor, sample_evenly: bool):
        """
        Given uncertainty map or a list of uncertainty maps, budget, and sample_evenly,
        return the top K pixels to label according to the budget and `sample_threshold`
        Returns (value, index) using torch.Tensor.topk API.
        """

        top_k_percent = None
        unit = reduce(lambda x, y: x*y, uc_map.shape)

        # define `selection_pool` -- number of pixels that the budget allows to select. in the case where
        # sample_threshold is set, selection_pool will be equal to the `top_k_percent` instead of the budget
        if hasattr(self.sample_settings, 'sample_threshold'):
            top_k_percent = self.sample_settings['sample_threshold']
            assert top_k_percent > 0., \
                f"Can only sample with sample_threshold > 0, but received {top_k_percent}"
            selection_pool = int(top_k_percent / 100 * unit)
        else:
            if sample_evenly:
                selection_pool = budget // self.query_dataset_size
            else:
                selection_pool = budget

        # define indices
        if self.sample_mode == 'region':
            indices = self.get_regions(selection_pool, uc_map)
        else:
            indices = self.get_pixels(selection_pool, uc_map)
       
        # when sample_threshold = True
        if top_k_percent != None:  
            if sample_evenly:
                query_size = budget // self.query_dataset_size
                if query_size == 0:
                    self.logger.warning("query_size is zero. check if the budget_per_round is correct.")
                indices = np.random.choice(indices, size=query_size, replace=False)
            else:
                assert len(indices) >= budget
                indices = np.random.choice(indices, size=budget, replace=False)
        indices_of_original_shape = np.unravel_index(indices, uc_map.shape)
        return indices_of_original_shape

    def extract_query_indices(self, uc_map, sample_evenly=True) -> np.ndarray:
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
            new_query:  A boolean mask of uc_map.shape indicating which pixels to label
        """

        assert (sample_evenly and len(uc_map.shape) == 2) or \
            (not sample_evenly and len(uc_map.shape) == 3)

        budget = self.sample_settings.budget_per_round
        uc_map_cuda = torch.FloatTensor(uc_map).cuda() 
        indices = self.query_by_budget(budget, uc_map_cuda, sample_evenly)
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
