"""
Customized functions from MMSeg
"""

import random
import torch 
from active.dataset import ActiveLearningDataset

def train_al_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):

    """
    - dataset (ActiveLearningDataset)
    - model (should be compatible / equivalent to nn.Module)
    
    Plan
    - Follow active learning steps in BAAL
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    if not torch.cuda.is_available():
        print("the experiement should only be run on gpu.")

    active_set = ActiveLearningDataset(dataset, pool_specifics=None)
    test_set = dataset
    active_set.label_randomly(cfg.active_learning.initial_pool)

    
    
