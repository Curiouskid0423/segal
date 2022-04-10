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
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from array_utils import stack_in_memory
from metrics import Loss

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

    def __init__(self, model, criterion, logger):
        self.model = model
        self.criterion = criterion
        self.logger = logger    # Use the same logger as the main file (MMSeg API)
        self.metrics = dict()   # Could have many metrics, not just train / test loss
        self.add_metric("loss", lambda: Loss())

    def add_metric(self, name: str, initializer: Callable):
        """
        Add Metric to the Model.

        Args:
            name (str): name of the metric.
            initializer (Callable): lambda to initialize a new instance of a
                                    baal.utils.metrics.Metric object.
        """
        self.metrics["test_" + name] = initializer()
        self.metrics["train_" + name] = initializer()

    def reset_metrics(self, filter=""):
        """
        Reset all Metrics according to a filter.

        Args:
            filter (str): Only keep the metric if `filter` in the name.
        """
        for k, v in self.metrics.items():
            if filter in k:
                v.reset()

    def update_metrics(self, out, target, loss, filter=""):
        """
        Update all metrics.

        Args:
            out (Tensor): Prediction.
            target (Tensor): Ground truth.
            loss (Tensor): Loss from the criterion.
            filter (str): Only update metrics according to this filter.
        """
        for k, v in self.metrics.items():
            if filter in k:
                if "loss" in k:
                    v.update(loss)
                else:
                    v.update(out, target)

    def train_on_dataset(
        self, dataset, optimizer, batch_size, epoch, use_cuda, workers=4,
        collate_fn: Optional[Callable] = None, regularizer: Optional[Callable] = None,
    ):
        """
        Train for `epoch` epochs on a Dataset `dataset.

        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.
            optimizer (optim.Optimizer): Optimizer to use.
            batch_size (int): The batch size used in the DataLoader.
            epoch (int): Number of epoch to train for.
            use_cuda (bool): Use cuda or not.
            workers (int): Number of workers for the multiprocessing.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.

        Returns:
            The training history.
        """
        self.train()
        history = []
        self.logger.info("Starting training", epoch=epoch, dataset=len(dataset))
        collate_fn = collate_fn or default_collate

        for _ in range(epoch):
            self.reset_metrics("train")
            for data, target in DataLoader(
                dataset, batch_size, True, num_workers=workers, collate_fn=collate_fn
            ):
                _ = self.train_on_batch(data, target, optimizer, use_cuda, regularizer)
            history.append(self.metrics["train_loss"].value)

        optimizer.zero_grad() 
        self.logger.info("Training complete", train_loss=self.metrics["train_loss"].value)
        return history

    def test_on_dataset(
        self, dataset, batch_size, use_cuda, workers = 4, 
        collate_fn: Optional[Callable] = None,
        average_predictions = 1,
    ):
        """
        Test the model on a Dataset `dataset`.

        Args:
            dataset (Dataset): Dataset to evaluate on.
            batch_size (int): Batch size used for evaluation.
            use_cuda (bool): Use Cuda or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            average_predictions (int): The number of predictions to average to
                compute the test loss.

        Returns:
            Average loss value over the dataset.
        """
        self.eval()
        self.logger.info("Starting evaluating", dataset=len(dataset))
        self.reset_metrics("test")

        for data, target in DataLoader(
            dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn
        ):
            _ = self.test_on_batch(
                data, target, cuda=use_cuda, average_predictions=average_predictions
            )

        self.logger.info("Evaluation complete", test_loss=self.metrics["test_loss"].value)
        return self.metrics["test_loss"].value

    def train_and_test_on_datasets(
        self, train_dataset, test_dataset, optimizer, batch_size, 
        epoch, use_cuda, workers = 4, return_best_weights=False,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
        patience=None,
        min_epoch_for_es=0,
    ):
        """
        Train and test the model on both Dataset `train_dataset`, 
        and `test_dataset` in each epoch

        Args:
            train_dataset:  Dataset to train on.
            test_dataset:   Dataset to evaluate on.
            optimizer:      Optimizer to use during training.
            batch_size:     Batch size used.
            epoch:          Number of epoch to train on.
            use_cuda:       Use Cuda or not.
            workers:        Number of workers to use (multi-processing)
            collate_fn:     The collate function to use.
            regularizer:    The loss regularization for training.
            return_best_weights: If True, will keep the best weights and return them.
            patience:       Hyperparameter for early stopping
            min_epoch_for_es:    Epoch at which the early stopping starts.

        Returns:
            History and best weights if required.
        """
        best_weight = None
        best_loss = 1e10
        best_epoch = 0
        hist = []
        for e in range(epoch):
            _ = self.train_on_dataset(
                train_dataset, optimizer, batch_size, 1, use_cuda, workers, collate_fn, regularizer
            )
            te_loss = self.test_on_dataset(test_dataset, batch_size, use_cuda, workers, collate_fn)
            hist.append({k: v.value for k, v in self.metrics.items()})
            if te_loss < best_loss:
                best_epoch, best_loss = e, te_loss
                if return_best_weights:
                    best_weight = deepcopy(self.state_dict())

            if patience is not None and (e - best_epoch) > patience and (e > min_epoch_for_es):
                # Early stopping
                break

        if return_best_weights:
            return hist, best_weight
        else:
            return hist

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
        if len(dataset) == 0:
            return None

        self.logger.info("Start Predict", dataset=len(dataset))
        collate_fn = collate_fn or default_collate

        loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        
        if verbose:
            loader = tqdm(loader, total=len(loader), file=sys.stdout)
        
        for idx, (data, _) in enumerate(loader):

            pred = self.predict_on_batch(data, iterations, use_cuda)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

    def predict_on_dataset(
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
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(
            self.predict_on_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                use_cuda=use_cuda,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def train_on_batch(
        self, data, target, optimizer, cuda=False, regularizer: Optional[Callable] = None
    ):
        """
        Train the current model on a batch using `optimizer`.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            optimizer (optim.Optimizer): An optimizer.
            cuda (bool): Use CUDA or not.
            regularizer (Optional[Callable]): The loss regularization for training.


        Returns:
            Tensor, the loss computed from the criterion.
        """

        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        self.update_metrics(output, target, loss, filter="train")
        return loss

    def test_on_batch(
        self, data, target, cuda = False, average_predictions = 1,
    ):
        """
        Test the current model on a batch.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            cuda (bool): Use CUDA or not.
            average_predictions (int): The number of predictions to average to
                compute the test loss.

        Returns:
            Tensor, the loss computed from the criterion.
        """
        with torch.no_grad():
            if cuda:
                data, target = data.cuda(), target.cuda()

            preds = map_on_tensor(
                lambda p: p.mean(-1),
                self.predict_on_batch(data, iterations=average_predictions, cuda=cuda),
            )
            loss = self.criterion(preds, target)
            self.update_metrics(preds, target, loss, "test")
            return loss

    def predict_on_batch(self, data, iterations=1, cuda=False):
        """
        Get the model's prediction on a batch.

        Args:
            data (Tensor): The model input.
            iterations (int): Number of prediction to perform.
            cuda (bool): Use CUDA or not.

        Returns:
            Tensor, the loss computed from the criterion.
            shape = {batch_size, nclass, n_iteration}.

        """
        with torch.no_grad():
            if cuda:
                data = data.cuda()
            data = map_on_tensor(lambda d: stack_in_memory(d, iterations), data)
            try:
                out = self.model(data)
            except RuntimeError as e:
                raise RuntimeError(
                    """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
                Use `replicate_in_memory=False` in order to reduce the memory requirements.
                Note that there will be some speed trade-offs"""
                ) from e
            out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
            out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)

            return out

    def get_params(self):
        """
        Return the parameters to optimize.
        """
        return self.model.parameters()

    def state_dict(self):
        """Get the state dict(s)."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load the model with `state_dict`."""
        self.model.load_state_dict(state_dict, strict=strict)

    def train(self):
        """Set the model in `train` mode."""
        self.model.train()

    def eval(self):
        """Set the model in `eval mode`."""
        self.model.eval()