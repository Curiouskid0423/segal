"""
Utility file for arrays (from BAAL)
"""
import torch

def stack_in_memory(data, iterations):
    """
    Stack `data` `iterations` times on the batch axis.
    Args:
        data (Tensor):  Data to stack
        iterations:     Number of time to stack.

    Returns:
        Tensor with shape [batch_size * iterations, ...]
    """
    input_shape = data.size()
    batch_size = input_shape[0]
    try:
        data = torch.stack([data] * iterations)
    except RuntimeError as e:
        raise RuntimeError(
            """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
        Use `replicate_in_memory=False` in order to reduce the memory requirements.
        Note that there will be some speed trade-offs"""
        ) from e
    data = data.view(batch_size * iterations, *input_shape[1:])
    return data
