# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader

def convert_vit(ckpt):

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        original = k[:]

        k = k.replace('blocks', 'layers')   
        k = k.replace('norm', 'ln') 
        # patch embedding
        k = k.replace('patch_embed.proj', 'patch_embed.projection')
        # self-attention
        k = k.replace('attn.qkv.weight', 'attn.attn.in_proj_weight')
        k = k.replace('attn.qkv.bias', 'attn.attn.in_proj_bias')
        # output projection
        k = k.replace('attn.proj.weight', 'attn.attn.out_proj.weight')
        k = k.replace('attn.proj.bias', 'attn.attn.out_proj.bias')
        # mlp
        k = k.replace('mlp.fc1', 'ffn.layers.0.0')
        k = k.replace('mlp.fc2', 'ffn.layers.1')
        # layer norms
        k = k.replace('ln.weight', 'ln1.weight')
        k = k.replace('ln.bias', 'ln1.bias')
        
        new_ckpt[k] = v
        print(f"{original} => {k}")

    return new_ckpt

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('--src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('--dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)

    # for k in weight.keys():
    #     print(k)
    
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)



if __name__ == '__main__':
    main()