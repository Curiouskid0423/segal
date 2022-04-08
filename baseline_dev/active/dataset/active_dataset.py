"""
ActiveLearningDataset is a wrapper class around a normal 
torchdata.Dataset class to enable "iteratively adding new 
samples from the unlabeled pool".

"""
import torch.utils.data as torchdata
from typing import Optional, Callable
import numpy as np

def _identity(x):
    return x

class ActiveLearningDataset:
    """
    Args:
        dataset: The baseline dataset, type = torchdata.Dataset
        labelled: 
            An array that acts as a mask which is greater than 1 for every data point 
            that is labelled, and 0 for every data point that is not labelled.
        make_unlabelled: 
            The function that returns an unlabelled version of a datum so that 
            it can still be used in the DataLoader.
        random_state: Set the random seed for label_randomly().
        pool_specifics: 
            Attributes to set when creating the pool. Useful to remove data augmentation.
        last_active_steps: 
            If specified, will iterate over the last_active_steps 
            instead of the full dataset. Useful when doing partial finetuning.
    """

    def __init__(self,
        dataset: torchdata.Dataset,
        labelled: Optional[np.ndarray] = None,
        make_unlabelled: Callable = _identity,
        random_state=None,
        pool_specifics: Optional[dict] = None,
        last_active_steps: int = -1,
        ):
        pass


    def label(self):
        pass