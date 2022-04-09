# Plan of Development

BAAL has any basic building blocks already. Read the code and adapt it 
such that MMSeg backbones may use seamlessly. The goal of this baseline system
is to enable the wide variety of MMSeg backabones to perform active learning.

- Expected result
  - Running your baseline system should involve similar procedure to running MMSeg, meaning that
  users should be able to simply change the config and run everything. Files to modify includes: 
  all Dataset-related files, make train_segmentor iteratively label the unlabeled-pool as in 
  `ActiveLearningLoop.step()`.

- ActiveLearningDataset
  - In baal, this is simply a 1-layer wrapper around a `torchdata.Dataset` class.
  - In MMSeg, the call stack from main() to constructing a dataloader is the following:
  `train_segmentor` > `build_dataloader` (which takes in a torchdata.Dataset and returns a torch.DataLoader,
  the Dataset class is passed in at the train_segmentor call.)
  - Stick with "image" level acquisition for now, but your system's value should be in 
  (1) enabling "pixel" or "region" level acquisition and (2) integration with MMSeg.
- Heursitics: add `AL-RIPU` with your own implementation

- Implement `ReAL` and test if it works on semantic segmentation!

----------------

wrapper = ModelWrapper(model, ...)
>> `wrapper.model` is of type "nn.Module". Pass this into the typical MMSeg pipeline.
al_dataset = ActiveLearningDataset(dataset, test_transform = {...})
>> `al_dataset.dataset` should be a torchdata.Subset that you can specify index with.
Simply append new index on it every time.


- Take in all the arguments
- Run (customized) train_segmentor
- Inside train_segmentor, we cannot define a static dataloader at the beginning because our we will be modifying the Dataset class (specifically, via appending more indices onto the indices list passed into torchdata.Subset), thus every epoch will need to instantiate a new DataLoader with our updated torchdata.Subset


If we want to avoid changes in MMCV, we might have to: for every epoch, call `runner.run(<static dataloader>, cfg.workflow)` with the currently-available labeled dataloader; somehow keep the state_dict (might not always need it but need to have this implemented); update the indices of torchdata.Subset and create a corresponding DataLoader; pass this new DataLoader to runner to initiate another epoch.
