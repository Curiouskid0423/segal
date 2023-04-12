"""
Code adapted from UC Berkeley CS182 Mask Autoencoder assignment
"""
import torch
import einops

def index_sequence(x: torch.Tensor, ids):
    """Index tensor (x) with indices given by ids
    Args:
        x: input sequence tensor, can be 2D (batch x length) or 3D (batch x length x feature)
        ids: 2D indices (batch x length) for re-indexing the sequence tensor
    """
    if len(x.shape) == 3:
        ids = ids.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.take_along_dim(x, ids, dim=1)

def random_masking(x, keep_length, ids_shuffle, channels=3):
    """Apply random masking on input tensor
    Args:
        x: input patches (batch x length x feature)
        keep_length: length of unmasked patches
        ids_shuffle: random indices for shuffling the input sequence 
    Returns:
        kept: unmasked part of x
        mask: a 2D (batch x length) mask tensor of 0s and 1s indicated which
            part of x is masked out. The value 0 indicates not masked and 1
            indicates masked.
        ids_restore: indices to restore x. If we take the kept part and masked
            part of x, concatentate them together and index it with ids_restore,
            we should get x back.
    """
    batch, length, feature_size = x.size()
    shuffled = index_sequence(x, ids_shuffle)
    
    kept = shuffled[:, :keep_length, :]
    ids_restore = torch.empty_like(ids_shuffle)
    for idx, row in enumerate(ids_shuffle):
        ids_restore[idx] = torch.argsort(row)
    mask = torch.empty(batch, length)
    for i in range(batch):
        mask[i] = torch.where(ids_restore[i] >= keep_length, 1., 0.)

    return kept.cuda(), mask.cuda(), ids_restore

def restore_masked(kept_x, masked_x, ids_restore):
    """
    Restore masked patches.
    Args:
        kept_x: unmasked patches
        masked_x: masked patches
        ids_restore: indices to restore x
    Returns:
        restored patches
    """
    return index_sequence(torch.cat((kept_x, masked_x), dim=1), ids_restore)

def patchify(images: torch.Tensor, patch_size: int = 4):
    """
    Splitting images into patches.
    Args:
        images: Input tensor with size (batch, channels, height, width)
            Assume that image is square where height == width.
    Returns:
        A batch of image patches with size (
            batch, 
            (height / patch_size) * (width / patch_size), 
            channels * patch_size * patch_size)
    """
    pattern = 'b c (h ph) (w pw) -> b (h w) (ph pw c)'
    return einops.rearrange(
        images, 
        pattern=pattern, 
        ph=patch_size,
        pw=patch_size
    )

def unpatchify(patches: torch.Tensor, patch_size: int = 4):
    """
    Combining patches into images.
    """

    batch, channels, num_patches, feature_size = patches.size()
    h, w = int(num_patches**0.5), int(num_patches**0.5)
    pattern = 'b c (h w) (ph pw) -> b c (h ph) (w pw)'
    return einops.rearrange(
        patches, pattern=pattern, 
        ph=patch_size, pw=patch_size, h=h, w=w)

def get_shuffled_ids(batch_size, total_patches, device='cuda'):
    ids_shuffle = torch.argsort(
        torch.rand((batch_size, total_patches)).to(device), dim=1)
    return ids_shuffle