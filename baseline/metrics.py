import torch
import numpy as np

class Metrics:
    """
    metric is an abstract class.
    Args:
        average (bool): whether to average the values of metrics
                        that are calculated in several trials.
    """

    def __init__(self, average=True, **kwargs):
        self._average = average
        self.eps = 1e-20
        self.reset()
        self.result = torch.FloatTensor()

    def reset(self):
        """Reset the private values of the class."""
        raise NotImplementedError

    def update(self, output=None, target=None):
        """
        Main calculation of the metric which updated the private values respectively.

        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        raise NotImplementedError

    def calculate_result(self):
        """
        calculate the final values when the epoch/batch loop is finished.
        """
        raise NotImplementedError

    @property
    def value(self):
        """output the metric results (array shape) or averaging
        out over the results to output one single float number.

        Returns:
            result (np.array / float): final metric result

        """
        self.result = torch.FloatTensor(self.calculate_result())
        if self._average and self.result.numel() == self.result.size(0):
            return self.result.mean(0).cpu().numpy().item()
        elif self._average:
            return self.result.mean(0).cpu().numpy()
        else:
            return self.result.cpu().numpy()

    @property
    def standard_dev(self):
        """Return the standard deviation of the metric."""
        result = torch.FloatTensor(self.calculate_result())
        if result.numel() == result.size(0):
            return result.std(0).cpu().numpy().item()
        else:
            return result.std(0).cpu().numpy()

    def __str__(self):
        val = self.value
        std = self.standard_dev
        if isinstance(val, np.ndarray):
            return ", ".join(f"{v:.3f}±{s:.3f}" for v, s in zip(val, std))
        else:
            return f"{val:.3f}±{std:.3f}"


class Loss(Metrics):
    """
    Args:
        average (bool): whether to average the values of metrics
                        that are calculated in several trials.
    """

    def __init__(self, average=True, **kwargs):
        super().__init__(average=average)

    def reset(self):
        self.loss = []

    def update(self, output=None, target=None):
        self.loss.append(output)

    def calculate_result(self):
        return self.loss