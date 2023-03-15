
from collections.abc import Sequence
import pickle
import torch
import torch.distributed as dist
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info
# from torch.utils.data.dataloader import default_collate
# from mmcv.engine import collect_results_cpu

def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [fn(v) for v in val]
    elif isinstance(val, dict):
        return {k: fn(v) for k, v in val.items()}
    return fn(val)


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result is not None:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

def batch_preprocess(data_batch):
    """ Preprocess data batch for type compatibility """

    if isinstance(data_batch['img'], DataContainer):
        ext_img = data_batch['img'].data[0].cuda()
    else:
        ext_img = data_batch['img'][0].data[0].cuda()

    if isinstance(data_batch['img_metas'], DataContainer):
        ext_img_meta = data_batch['img_metas'].data[0]
    else:
        ext_img_meta = data_batch['img_metas'][0]

    return ext_img, ext_img_meta