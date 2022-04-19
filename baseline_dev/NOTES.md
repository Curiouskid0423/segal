# Plan of Development

- Expected result
  - Running your baseline system should involve similar procedure to running MMSeg, meaning that
  users should be able to simply change the config and run everything. Files to modify includes: 
  all Dataset-related files, make train_segmentor iteratively label the unlabeled-pool as in 
  `ActiveLearningLoop.step()`.
  - `bash <script> <config_file> <GPU_number>`

- Sticking with "image" level acquisition for now, but your system's value should be in 
  (1) enabling "pixel" or "region" level acquisition and (2) integration with MMSeg.
- Heursitics: add `AL-RIPU` with your own implementation

- Implement `ReAL` and test if it works on semantic segmentation.

- FIXME: `metrics.py` doesn't seem to be needed anymore.

----------------
