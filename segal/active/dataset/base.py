""" OracleDataset: 
An Abstract Dataset that contains some generic methods
useful in Active Learning for any modality. This dataset is called 
"SplittedDataset" in baal, but renamed here for consistency 
with its instance methods.
"""
import numpy as np
import torch.utils.data as torchdata
from sklearn.utils import check_random_state

class OracleDataset(torchdata.Dataset):

    """ Abstract Dataset class that contains useful API calls.

    labelled: 
        List with mask values. If value > 1 (= its active step), 
        the data is labelled and vice versa.
    random_state: 
        Set the random seed for label_randomly().
    last_active_steps (kept from baal module):
        For fine-tuning. If specified, will iterate over the last active steps
        instead of the full dataset. For example if the complete AL training recipe is
        5 steps but you only want to fine-tune for the last 3 steps, set this to 3.
    """

    def __init__(self, labelled, random_state=None, last_active_steps=-1):

        self.labelled_map = labelled
        self.random_state = check_random_state(random_state)
        self.last_active_steps = last_active_steps

    @property
    def labelled(self):
        """Return labelled map but in boolean format"""
        return self.labelled_map.astype(bool)

    @property
    def curr_al_step(self):
        """Get the current active learning step."""
        return int(self.labelled_map.max())

    @property
    def num_labelled(self):
        """Number of labelled data points."""
        return self.labelled.sum()

    @property
    def num_unlabelled(self):
        """Number of unlabelled data points."""
        return (~self.labelled).sum()

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        
        """Return the number of labeled data pairs available currently"""

        return len(self.get_indices_for_active_step())

    def get_indices_for_active_step(self):
        
        """
        Return list of selected indices for training
        at the current time step / active learning step.
        """
        # when `last_active_steps = K`, we will only be getting indices after round K
        if self.last_active_steps == -1:
            threshold = 0
        else:
            threshold =  max(0, self.current_al_step - self.last_active_steps)

        indices = [index for index, val in enumerate(self.labelled_map) if val > threshold]
        return indices

    def is_labelled(self, index):
        """
        Return true is the given index is labelled
        """
        return self.labelled_map[index] > 0

    def label(self, index):
        """
        Abstract method
        Label the pool with the given indices. The index should be relative
        to the pool, not the overall OracleDataset.
        Args:
            index: One or many indices to be labelled.
        """
        return NotImplementedError

    def label_randomly(self, n = 1):

        """
        Label n random points from the pool.
        """

        to_label = list(self.random_state.choice(self.num_unlabelled, size=n, replace=False))
        self.label(to_label)

    def _labelled_to_oracle_index(self, index):
        recovered_list = self.labelled.nonzero()[0]
        return int(recovered_list[index].squeeze().item())

    def _pool_to_oracle_index(self, index):
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        recovered_list = (~self.labelled).nonzero()[0]
        return [int(recovered_list[idx].squeeze().item()) for idx in index]

    def _oracle_to_pool_index(self, index):
        if isinstance(index, int):
            index = [index]

        # Pool indices are the unlabelled, starts at 0
        recovered_list = np.cumsum(~self.labelled) - 1
        return [int(recovered_list[idx].squeeze().item()) for idx in index]
