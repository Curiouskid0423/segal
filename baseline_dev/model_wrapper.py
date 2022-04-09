"""
ModelWrapper to contain an instance attribute of 
type torch.nn.Module that MMSeg can work on. 
"""

from typing import Callable
import torch.nn as nn


class ModelWrapper:

    def __init__(self, model: nn.Module, criterion: Callable):

        self.model = model
        self.criterion = criterion

    def train_on_dataset(self,):
        pass        
        