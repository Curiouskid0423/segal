"""
ActiveLearningDataset is a wrapper class around a normal 
torchdata.Dataset class to enable "iteratively adding new 
samples from the unlabeled pool".

ActiveLearningPool is a normal CustomDataset from torch 
for unlabeled dataset.
"""
from itertools import zip_longest
import torch.utils.data as torchdata
from typing import Optional, Callable
import numpy as np
from copy import deepcopy
from active.dataset.base import OracleDataset

def _identity(x):
    return x

class ActiveLearningDataset(OracleDataset):
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
        
        self.dataset = dataset
        if labelled is not None:
            self.labelled_map = labelled.astype(int)
        else:
            self.labelled_map = np.zeros(len(dataset), dtype=int)
        self.pool_specifics = {} if pool_specifics is None else pool_specifics
        self.make_unlabelled = make_unlabelled
        self.can_label = self.check_can_label()
        # Constructor of OracleDataset
        super().__init__(self.labelled_map, random_state, last_active_steps)


    def __getitem__(self, index):
        # index should be relative to the currently available list of indices
        index = self.get_indices_for_active_step()[index]
        return self.dataset[index]
    
    class ActiveIter:
        
        """Need an iterator class for ActiveLearningDataset"""
        def __init__(self, aldataset):
            self.index = 0
            self.aldataset = aldataset

        def __len__(self):
            return len(self.aldataset)

        def __next__(self):
            if self.index >= len(self):
                raise StopIteration

            n = self.aldataset[self.index]
            self.index += 1
            return n

    def __iter__(self):
        """Return an iterator instance"""
        return self.ActiveIter(self)

    @property
    def pool(self):
        """Returns a new Dataset made from unlabelled samples"""
        # Copy the OracleDataset
        pool_dataset = deepcopy(self.dataset)
        # Apply test transform
        for attr, value in self.pool_specifics.items():
            if hasattr(pool_dataset, attr):
                setattr(pool_dataset, attr, value)
            else:
                raise ValueError(f"{pool_dataset} doesn't have {attr}")

        # Exclude the labelled data
        recovered_index = (~self.labelled).nonzero()[0].flatten()
        pool_dataset = torchdata.Subset(pool_dataset, list(recovered_index))
        res = ActiveLearningPool(pool_dataset, make_unlabelled=self.make_unlabelled)
        return res

    def check_can_label(self):
        if hasattr(self.dataset, "label") and callable(self.dataset.label):
            return True
        return False

    def label(self, index, value=None):
        """
        Update the list of indices with heuristically selected samples.
        Index is relative to pool, not the overall OracleDataset
        Args:
            index: one or more indices to be labelled
            value: when provided, set corresponding index to the given value(s),
                   otherwise just use the original dataset's label(s)
        """
        index_list = [index] if isinstance(index, int) else index
        value_list = [value] if (value is None) or isinstance(value, int) else value
        if (value_list[0] is not None) and len(index_list) != len(value_list):
            raise ValueError("Index list and Value list does not match in ActiveLearningDataset")

        indices = self._pool_to_oracle_index(index_list)
        curr_active_step = self.curr_al_step + 1

        for index, val in zip_longest(indices, value_list, fillvalue=None):
            if not self.can_label:
                self.labelled_map[index] = curr_active_step
            elif val is not None:
                self.dataset.label(index, val)
                self.labelled_map[index] = curr_active_step
            elif val is None:
                raise ValueError("The dataset is able to label data, but no label was provided.")

    def reset_labelled(self):
        """Reset the label map."""
        self.labelled_map = np.zeros(len(self.dataset), dtype=np.bool)

    def get_raw(self, idx: int):
        """Get a datapoint from the underlying dataset."""
        return self.dataset[idx]
    
    def state_dict(self):
        """Return the state_dict, ie. the labelled map and random_state."""
        return {"labelled": self.labelled_map, "random_state": self.random_state}

    def load_state_dict(self, state_dict):
        """Load the labelled map and random_state with give state_dict."""
        self.labelled_map = state_dict["labelled"]
        self.random_state = state_dict["random_state"]


class ActiveLearningPool(torchdata.Dataset):
    """
    A dataset that represents the unlabelled pool for active learning.
    """
    def __init__(
        self, dataset: torchdata.Dataset, make_unlabelled: Callable = _identity):
        self.dataset = dataset
        self.make_unlabelled = make_unlabelled
    
    def __getitem__(self, index):
        return self.make_unlabelled(self.dataset[index])

    def __len__(self):
        return len(self.dataset)