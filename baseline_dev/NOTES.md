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
- Heursitics 
  - Start with simple ones that baal has implemented such as `bald` or `entropy`
  - Add `AL-RIPU` with your own implementation

- Implement `ReAL` and test if it works on semantic segmentation!