# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, master_only, WandbLoggerHook

@HOOKS.register_module()
class WandbLoggerHookWithVal(WandbLoggerHook):

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            tags['global_step'] = self.get_iter(runner)
            if self.with_step:
                is_val = False
                for key in tags:
                    if "val/" in key:
                    # if currently reporting the val metric
                        is_val = True
                        break
                if is_val:
                    tags['global_step'] += 1
                self.wandb.log(
                    tags, step=tags['global_step'],
                    commit=False if is_val else self.commit)
            else:
                # tags['iter'] = self.get_iter(runner)
                # tags['epoch'] = self.get_epoch(runner)
                self.wandb.log(tags, commit=self.commit)
