"""
Customized runner file for running MAE multitask active learning
"""

from mmcv.runner.builder import RUNNERS

from segal.runner import ActiveLearningRunner

@RUNNERS.register_module()
class MultiTaskActiveRunner(ActiveLearningRunner):

    """
    Override ActiveLearningRunner to be compatible for multi-task learning 
    without affecting previously working code on active domain adaptation
    """

    def __init__(
        self, model, batch_processor=None, optimizer=None, work_dir=None, logger=None, 
        meta=None, max_iters=None, max_epochs=None, sample_mode=None, sample_rounds=None):
        super(ActiveLearningRunner, self).__init__(
            model, batch_processor, optimizer, work_dir, logger, meta, 
            max_iters, max_epochs, sample_mode, sample_rounds)


    # def train(self, data_loader, **kwargs):
    #     pass

    # def val(self, data_loader, **kwargs):
    #     pass

    # def query(self, data_loader, **kwargs):
    #     pass

    # def run_iters(self):
    #     pass

    # def run(self):
    #     # main function
    #     """
        
    #     setup_active
    #     while epoch < total_epochs:
    #         self.forward()
    #         self.optimizer_seg.zero_grad()
    #     """