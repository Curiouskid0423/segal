import numpy as np
import warnings
import torch
from torch import Tensor
import torch.nn.functional as F
from collections.abc import Sequence
from segal.utils.heuristics import to_prob, RIPU_Net

available_reductions = {
    "max": lambda x: np.max(x, axis=tuple(range(1, x.ndim))),
    "min": lambda x: np.min(x, axis=tuple(range(1, x.ndim))),
    "mean": lambda x: np.mean(x, axis=tuple(range(1, x.ndim))),
    "sum": lambda x: np.sum(x, axis=tuple(range(1, x.ndim))),
    "none": lambda x: x,
}

def _shuffle_subset(data: np.ndarray, shuffle_prop: float) -> np.ndarray:
    to_shuffle = np.nonzero(np.random.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[np.random.permutation(to_shuffle), ...]
    return data


class AbstractHeuristic:
    
    """
    Abstract class that defines a Heuristic with template methods 
    such as get_ranks, compute_score

    Args:
        shuffle_prop (float): shuffle proportion.
        reverse (bool): True if the most uncertain sample has the highest value, and vice versa.
        reduction (Union[str, Callable]): Reduction used after computing the score.
    """

    def __init__(self, shuffle_prop=0.0, reverse=False, reduction="none"):
        # self.shuffle_prop = shuffle_prop
        self.reversed = reverse
        assert reduction in available_reductions or callable(reduction)
        self._reduction_name = reduction
        self.reduction = reduction if callable(reduction) else available_reductions[reduction]

    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.
        To be implemented by Child class.
        """
        raise NotImplementedError

    def get_uncertainties_generator(self, predictions):
        """
        Compute the score according to the heuristic.
        Can be used in CombineHeuristics.
        """
        raise NotImplementedError

    def get_uncertainties(self, predictions, **kwargs):

        """ 
        Get the uncertainties.
        Args:   Array of predictions (Tensor)
        Return: Array of uncertainties
        """

        scores = self.compute_score(predictions, **kwargs)
        # scores = self.reduction(scores)
        
        if not np.isfinite(scores).all():
            fixed = 0.0 if self.reversed else 10000 
            warnings.warn("Some value in scores vector is infinite.")
            scores[~np.isfinite(scores)] = fixed

        return scores

    def reorder_indices(self, scores):
        """
        Re-order indices given their uncertainty score.

        Args:
            scores (ndarray/ List[ndarray]): Array of uncertainties or list of arrays.

        Returns:
            ordered index according to the uncertainty (highest to lowes).

        Raises:
            ValueError if `scores` is not uni-dimensional.
        """
        if isinstance(scores, Sequence):
            scores = np.concatenate(scores)

        if scores.ndim > 1:
            raise ValueError(
                (
                    f"Can't order sequence with more than 1 dimension."
                    f"Currently {scores.ndim} dimensions."
                )
            )
        assert scores.ndim == 1  # We want the uncertainty value per sample.
        ranks = np.argsort(scores) # Ascending order
        if self.reversed:
            ranks = ranks[::-1] # Descending order
        # ranks = _shuffle_subset(ranks, self.shuffle_prop)
        return ranks

    def get_ranks(self, predictions):
        """
        Rank the predictions according to their uncertainties.
        Returns:
            Ranked index according to the uncertainty (highest to lowes).
            Scores for all predictions.
        """
        
        scores = self.get_uncertainties(predictions)

        return self.reorder_indices(scores), scores

    def __call__(self, predictions):
        """Rank according to their uncertainties.
        Only return the scores and not the associated uncertainties.
        """
        return self.get_ranks(predictions)[0]

class Random(AbstractHeuristic):

    def __init__(self, mode="image", seed=None):
        super().__init__(reverse=False)

        self.mode = mode
        # rng = random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def compute_score(self, predictions, **kwargs):
        
        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()
        
        if self.mode == 'image':
            return self.rng.rand(predictions.shape[0])
        elif self.mode in ['pixel', 'region']:
            l, c, h, w = predictions.shape
            return self.rng.rand(l, h, w).astype(dtype=np.float16)
        else:
            module_name = self.__class__.__name__
            raise NotImplementedError(f"mode {self.mode} not supported in {module_name}")

class Entropy(AbstractHeuristic):
    """
    Sort by entropy. The higher, the more uncertain.
    """
    def __init__(self, mode):
        # reverse = True to turn "ascending" result into a descending order.
        super().__init__(reverse=True)
        self.mode = mode
    
    def get_entropy(self, p, dim, keepdim):
        return torch.sum(-p * torch.log(p + 1e-6), dim=dim, keepdim=keepdim)

    def pixel_mean_entropy(self, softmax_pred):
        bs, classes, img_H, img_W = softmax_pred.size()
        entropy_map = self.get_entropy(softmax_pred, dim=1, keepdim=False)
        entropy_map = entropy_map.reshape(shape=(bs, img_H, img_W))
        return entropy_map.cpu().numpy()

    def image_mean_entropy(self, softmax_pred):
        bs, classes, img_H, img_W = softmax_pred.size()
        flat_probs = softmax_pred.reshape(bs, -1)
        entropy_lst = self.get_entropy(flat_probs, dim=-1, keepdim=False)
        return entropy_lst.cpu().numpy()

    def compute_score(self, predictions, **kwargs):
        """ `predictions` have to be a Tensor """
        probs = to_prob(predictions)
        if self.mode == 'image':
            return self.image_mean_entropy(probs)
        elif self.mode in ['pixel', 'region']:
            return self.pixel_mean_entropy(probs)
        else:
            module_name = self.__class__.__name__
            raise NotImplementedError(f"mode {self.mode} not supported in {module_name}")

class MarginSampling(AbstractHeuristic):
    """
    Sort by argmin(argmax(prob) - argmax2(prob)). The smaller the
    difference is, the more uncertain. Not that instead of maximizing,
    we want to minimize this value, as opposed to Entropy and Random
    """

    def __init__(self, mode):
        super().__init__(reverse=False)
        self.mode = mode

    def compute_score(self, predictions, **kwargs):

        softmax_pred = to_prob(predictions) # softmax_pred size: (b, c, h, w)

        assert isinstance(softmax_pred, torch.Tensor), "Predictions in MarginSampling has to be Tensor"
        queries = softmax_pred.topk(k=2, dim=1).values 
        query_map = (queries[:, 0, :, :] - queries[:, 1, :, :]).abs() # shape: (b, h, w)
        
        if self.mode == 'image':
            query_lst = query_map.reshape(query_map.shape[0], -1).mean(dim=-1) # shape: (b, 1)
            return query_lst.cpu().numpy()
        elif self.mode in ['pixel', 'region']:
            return query_map.cpu().numpy()
        else:
            module_name = self.__class__.__name__
            raise NotImplementedError(f"mode {self.mode} not supported in {module_name}")

class RegionImpurity(AbstractHeuristic):
    """
    Region Impurity loss from AL-RIPU paper https://arxiv.org/abs/2111.12940
    """

    def __init__(self, mode, categories, k=1, use_entropy=True):
        """
        Args:
            mode (str):     sampling mode. has to be either `pixel` or `region`
            k(int):         region size is defined as (2K+1) * (2K+1)
        """
        # reverse is set to true when the higher the uncertainty score, the more we want to sample it.
        super().__init__(reverse=True)
        self.mode = mode
        self.use_entropy = use_entropy
        self.ripu_net = RIPU_Net(size=2*k+1, channels=categories).cuda()

    def compute_score(self, predictions, **kwargs):
        """
        Args:
            prediction (np.array | Tensor): non-softmax logit score (size of [b, c, h, w])

        Return:
            uncertainty score (Tensor):     uncertainty score map of size [b, h, w]
        """

        softmax_pred = to_prob(predictions) # softmax_pred size: (b, c, h, w)
        assert isinstance(softmax_pred, torch.Tensor), "Predictions in Region-Impurity has to be Tensor"
        scores = self.ripu_net(softmax_pred, use_entropy=self.use_entropy) # should be (b, h, w)
        return scores


class MaskPredictionScore(AbstractHeuristic):
    """
    Use standardized MAE loss as sampling scores. 
    """

    def __init__(self, mode, crop_size, use_entropy=True, k=1, categories=19):
        """
        Args:
            mode (str):     sampling mode. has to be either `pixel` or `region`
            k(int):         region size is defined as (2K+1) * (2K+1)
        """
        # reverse is set to true when the higher the uncertainty score, 
        # the more we want to sample it, which is true for MAE loss.
        super().__init__(reverse=True)
        self.mode = mode
        self.use_entropy = use_entropy 
        if use_entropy:
            self.ripu_engine = RegionImpurity(mode='pixel', categories=categories, k=k)
        self.crop_size = crop_size
        assert len(crop_size)==2, 'crop_size has to be a 2-element tuple'
        self.stride = int(min(crop_size[0], crop_size[1]) * 0.5)

    def compute_score(self, network, image, seg_logit=None, mix_factor=1., just_mae=False):
        """
        Return:
            uncertainty score (Tensor):    uncertainty score map of size [b, h, w]
        """

        assert len(image)==1, "MAE score can only be computed one at a time"
        predictions = self.slide_inference(image, network)
        pixel_loss_map = (predictions - image) ** 2 * 0.5
        pixel_loss_map = pixel_loss_map.mean(dim=1).cpu().numpy() # [1, H, W]
        mae_score = pixel_loss_map / (pixel_loss_map.max() + 1e-6)
        if self.use_entropy and (not just_mae):
            assert seg_logit != None
            ripu_score = self.ripu_engine.get_uncertainties(seg_logit) # [1, H, W]
            ripu_score = ripu_score / (ripu_score.max() + 1e-6)
            mix_score = ripu_score * mix_factor + mae_score     # mix the two scores
            mix_score = mix_score / (mix_score.max() + 1e-6)    # normalize
            return mix_score
        else:
            return mae_score
    
    def slide_inference(self, image, network):
        """
        e.g. an 1280 x 640 image, with a MAE model trained on 384x384 crops
        the number of sliding inference will be 
        ceil((1280 - 384) / 192) + 1 = 6 on width
        ceil((640 - 384) / 192) + 1 = 3 on height
        18 inferences in total. note that we use "ceil" and not "floor" since all 
        pixels need to be computed.
        """
        assert isinstance(image, torch.Tensor) and len(image.shape)==4
        assert hasattr(network, 'mae_inference')
        batch, channels, img_h, img_w = image.shape
        crop_h, crop_w = self.crop_size

        h_grids = max(img_h-crop_h+self.stride-1, 0) // self.stride + 1
        w_grids = max(img_w-crop_w+self.stride-1, 0) // self.stride + 1

        result = torch.zeros_like(image)
        count = torch.zeros(size=(batch, 1, img_h, img_w))
        count = count.to(image.device)
        with torch.no_grad():
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    # calculate the starting pixels
                    start_y = h_idx * self.stride
                    start_x = w_idx * self.stride
                    # calculate the ending pixels
                    end_x = min(start_x + crop_w, img_w)
                    end_y = min(start_y + crop_h, img_h)
                    # adjust the start pixels based on the ending pixels
                    start_x = max(end_x - crop_w, 0)
                    start_y = max(end_y - crop_h, 0)
                    # inference
                    cropped_img = image[:, :, start_y:end_y, start_x:end_x]
                    cropped_pred = network.mae_inference(cropped_img)
                    # append results
                    # padding_amount = (start_x, img_w - end_x, start_y, img_h - end_y)
                    # result += F.pad(cropped_pred, pad=padding_amount)
                    result[:, :, start_y:end_y, start_x:end_x] += cropped_pred
                    count[:, :, start_y:end_y, start_x:end_x] += 1
        assert (count == 0).sum() == 0
        result = result / count
        return result