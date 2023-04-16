"""
util file for the new proposed method (multitask active learning)
"""
import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from mmcv.parallel  import DataContainer
from segal.utils.masking import unpatchify

tensor2image = ToPILImage(mode='RGB')

def get_crop_bbox(img, crop_size):
    """Randomly get a crop bounding box."""
    channels, height, width = img.shape
    margin_h = max(height - crop_size[0], 0)
    margin_w = max(width - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

def get_random_crops(image, crop_size, num=1):

    def crop(img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[..., crop_y1:crop_y2, crop_x1:crop_x2]
        return img

    results = []
    for i in range(num):
        crop_bbox = get_crop_bbox(image, crop_size=crop_size)
        results.append(crop(image, crop_bbox))
    return results


def save_reconstructed_image(
    path, ori: torch.Tensor, rec: torch.Tensor, img_metas: DataContainer, 
    index: int, overlay_visible_patches: bool, masks: torch.Tensor = None, patch_size=None):
    
    assert ori.shape == rec.shape
    assert isinstance(img_metas, DataContainer)
    _, H, W = ori.shape
    os.makedirs(name=path, exist_ok=True)
    norm_cfg = img_metas.data['img_norm_cfg']
    mean, std = norm_cfg['mean'], norm_cfg['std']
    mean, std = mean[:, None, None], std[:, None, None]

    unnormalize = lambda x: torch.clip((x.cpu() * std + mean).type(torch.uint8), 0, 255)
    file_name = osp.join(path, f'sample_{index}.jpg')

    if not overlay_visible_patches:
        result = Image.new('RGB', (W*2, H))
        result.paste(
            im=tensor2image(unnormalize(ori)), box=(0,0))
        result.paste(
            im=tensor2image(unnormalize(rec)), box=(W, 0))
    else:
        assert masks != None and patch_size != None

        # reshape masks (1 is removed, 0 is visible)
        masks = masks.unsqueeze(-1) # (batch, num_patches, 1)
        masks = masks.repeat(1, 1, patch_size**2 * 3) # (batch, num_patches, feature_len) 
        masks = unpatchify(masks, patch_size) # (batch, 3, img_height, img_width)
        assert len(masks)==1, "should be only one image"
        masks = masks.squeeze(0)

        ori = ori.cuda()
        ori_with_masks = ori * (1-masks)
        overlayed_rec = ori * (1-masks) + rec * masks

        result = Image.new('RGB', (W*3, H))
        result.paste(im=tensor2image(unnormalize(ori)), box=(0,0))
        result.paste(im=tensor2image(unnormalize(ori_with_masks)), box=(W,0))
        result.paste(im=tensor2image(unnormalize(overlayed_rec)), box=(2*W, 0))
        
    result.save(file_name)