"""
Customized functions from MMSeg
Equivalent to mmseg/apis/train.py
"""

import random
from copy import deepcopy
import torch 
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger

from active.dataset import ActiveLearningDataset
from baseline_dev.active import get_heuristics
from baseline_dev.active.active_loop import ActiveLearningLoop
from baseline_dev.model_wrapper import ModelWrapper

def train_al_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None,
                    logger=None):

    """
    - dataset (ActiveLearningDataset)
    - model (should be compatible / equivalent to nn.Module)
    
    Plan
    - Follow active learning steps in BAAL
    """
    if logger is None:
        logger = get_root_logger(cfg.log_level)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    if not torch.cuda.is_available():
        return ValueError("the experiement should only be run on gpu.")

    active_set = ActiveLearningDataset(dataset, pool_specifics=None)
    test_set = deepcopy(dataset)

    active_set.label_randomly(cfg.active_learning.initial_pool)

    heuristic = get_heuristics(
        cfg.active_learning['heuristic'], 
        cfg.active_learning['shuffle_prop'],
        )
    criterion = CrossEntropyLoss()
    model.cuda()
    # ignoring distributed mode for now
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    optimizer = build_optimizer(model, cfg.optimizer)

    model = ModelWrapper(model, criterion, logger)

    # FIXME: `samples_per_gpu` and `workers_per_gpu` should be used to 
    # distribute data on different GPU during training.
    batch_size = cfg.data['samples_per_gpu'] * cfg.data['workers_per_gpu'] * len(cfg.gpu_ids)
    
    active_loop = ActiveLearningLoop(
        dataset=active_set, 
        get_probabilities=model.predict_on_dataset,
        heuristic=heuristic,
        query_size=cfg.active_learning['query_size'],
        batch_size=batch_size,
        iterations=cfg.active_learning["iterations"],
        use_cuda=torch.cuda.is_available(),
        )


    # Following BAAL: Reset the weights at each active learning step.
    init_weights = deepcopy(model.state_dict())
    
    train_epochs = cfg.workflow[0][1]

    for epoch in range(train_epochs):

        # Load the initial weights.
        model.load_state_dict(init_weights)
        # Train
        model.train_on_dataset(
            active_set,
            optimizer,
            batch_size,
            cfg.active_learning["learning_epoch"],
            use_cuda=torch.cuda.is_available(),
        )

        # Validation
        model.test_on_dataset(test_set, batch_size, torch.cuda.is_available())
        metrics = model.metrics
        should_continue = active_loop.step()
        if not should_continue:
            break

        val_loss = metrics["test_loss"].value
        logs = {
            "val": val_loss,
            "epoch": epoch,
            "train": metrics["train_loss"].value,
            "labeled_data": active_set.labelled,
            "Next Training set size": len(active_set),
        }
        print(logs)