from logging import Logger
from typing import Union, List
from copy import deepcopy
from torch.utils.data import Dataset
from mmcv.utils import Config
from typing import Dict
from mmseg.datasets import build_dataset
from segal.dataset_wrapper import ConcatDataset
from segal.utils.train import flatten_nested_tuple_list as flatten, is_nested_tuple

def append_mask_dir(mask_dir: str, curr_dataset: Union[Config, List]) -> Config:
    """
    Ensure that mask_dir is correctly provided for appropriate cases
    Args:
        config:         the overall Config instance
        curr_dataset:   current dataset Config instance that requires mask_dir
    Return:
        a Config instance for the current dataset with mask_dir appended
    """


    if isinstance(curr_dataset, list):
        return [append_mask_dir(mask_dir, curr) for curr in curr_dataset]
    
    assert hasattr(curr_dataset, 'pipeline')
    new_config = deepcopy(curr_dataset)
    for idx, pipe in enumerate(new_config.pipeline):
        if 'LoadMasks' == pipe.type:
            setattr(new_config.pipeline[idx], "mask_dir", mask_dir)
            return new_config


def preprocess_datasets(config: Config, logger: Logger) -> Dict[str, Dataset]:

    datasets = {}
    is_active_learning =  config.runner.type in ['ActiveLearningRunner', 'MultiTaskActiveRunner']
    # preprocess workflow
    if is_active_learning:
        if is_nested_tuple(config.workflow[0]):
            assert len(config.workflow) == config.runner.sample_rounds+1, \
                "for irregular sampling, the number of outer tuples in `workflow` has to be equal sample_rounds"
    
        flow = flatten(config.workflow)

        assert all([mode in ['train', 'val', 'query'] for (mode, _) in flow]), \
            "workflow has to be either train, val, or query"
    else:
        flow = config.workflow

    # iterate pass the workflow
    for mode, _ in flow:
        data_cfg = getattr(config.data, mode)
        # no need to independently create a `query` dataset in `image-sampling`
        if mode == 'query' and config.runner.sample_mode == 'image':
            continue
        # make sure to use train pipeline cuz test pipeline does not load ground truth
        if mode == 'val':
            assert isinstance(data_cfg.pipeline, list)
            data_cfg.pipeline = config.data.train.pipeline
        # avoid creating multiple dataset for the same workflow (e.g. 'train', 'query')
        if not (mode in datasets.keys()):
            logger.info(f'Loading `{mode}` set...')
            
            if is_active_learning and (mode in ['train', 'query']):

                assert hasattr(config, 'mask_dir'), "`mask_dir` argument is required"
                data_cfg = append_mask_dir(config.mask_dir, data_cfg)

                if mode == 'train':
                    # case: train_set in either source-free or ADA setting
                    if config.source_free:
                        datasets[mode] = build_dataset(data_cfg, 
                                    dict(test_mode=False if mode=='train' else True))
                    else:
                        assert isinstance(data_cfg, list) and len(data_cfg)==2
                        source, target = data_cfg
                        source_set = build_dataset(source, dict(test_mode=False))
                        target_set = build_dataset(target, dict(test_mode=False))
                        datasets['source'], datasets['target'] = source_set, target_set
                        datasets['train'] = ConcatDataset([source_set, target_set], separate_eval=False)
                else:
                    # case: query_set in either source-free or ADA setting
                    datasets['query'] = build_dataset(data_cfg, dict(test_mode=False))

            else: 
                # case: (1) fully supervised learning 
                #       (2) val and test set in either source-free or ADA setting
                assert isinstance(data_cfg.pipeline, list)
                datasets[mode] = build_dataset(data_cfg, 
                                    dict(test_mode=False if mode=='train' else True))
    
    if is_active_learning and not config.source_free:
        LT, LQ = len(datasets['train']), len(datasets['query'])
        logger.info(f"concatenated query_set into the train_set. train_set size = {LT}, query_set size = {LQ}")

    return datasets

def insert_ignore_index(config: Config, value: int):
    assert hasattr(config, "model")
    for k in config.model.keys():
        entry = getattr(config.model, k)
        if isinstance(entry, dict) and 'loss_decode' in entry:
            entry.ignore_index = value
            entry.loss_decode.avg_non_ignore=True
