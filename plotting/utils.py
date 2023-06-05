from tqdm import tqdm 
import numpy as np
import torch
import torch.nn.functional as F
from mmseg.datasets import build_dataloader

def preprocess_data(data_container):
    img = data_container['img'].data[0]
    assert len(img.shape)==4 and len(img)==1
    img_metas =  data_container['img_metas']
    ground_truth = data_container['gt_semantic_seg'].data[0].squeeze(0)
    return img.cuda(), img_metas, ground_truth.cuda()

def onehot_with_ignore_label(labels, num_classes, ignore_label):
    dummy_label = num_classes + 1
    mask = (labels == ignore_label) # mask==1 should be removed
    modified_labels = labels.clone()
    modified_labels[mask] = num_classes
    # One-hot encode the modified labels
    one_hot_labels: torch.Tensor = F.one_hot(modified_labels, num_classes=dummy_label)
    # Remove the last row in the one-hot encoding
    one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)
    one_hot_labels = one_hot_labels[:, :-1, :, :]
    return one_hot_labels.to(float)


def compute_loss(model, data_container, loss_fn):
    img, img_metas, gt_semantic_seg = preprocess_data(data_container)
    ground_truth = onehot_with_ignore_label(gt_semantic_seg, num_classes=19, ignore_label=255)
    logits = model.whole_inference(img, img_metas, rescale=False) # [1, 19, 640, 1280]
    logits = logits.to(float)
    # print(f"ground_truth.shape: {ground_truth.shape}")
    # print(f"logits.shape: {logits.shape}")
    return loss_fn(logits, ground_truth, reduce='mean')

def get_seg_ranking(dataset, config, network, loss_fn, return_loss=False):
    network = network.cuda()
    SPG, WKR = config['samples_per_gpu'], config['workers_per_gpu']
    loader_cfg = dict(
        num_gpus=1, shuffle=False, dist=False, seed=0, 
        drop_last=False, samples_per_gpu=SPG, workers_per_gpu=WKR)
    data_loader = build_dataloader(dataset, **loader_cfg, ) 

    losses = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            curr_loss = compute_loss(network, batch, loss_fn=loss_fn)
            losses.append(curr_loss.cpu().numpy())

    losses = np.array(losses)
    indices = losses.argsort() # ascending order
    if return_loss:
        return indices, losses
    else:
        return indices

def get_mae_ranking(dataset, config, network,  mask_ratio=None, return_loss=False):
    network = network.cuda()
    SPG, WKR = config['samples_per_gpu'], config['workers_per_gpu']
    loader_cfg = dict(
        num_gpus=1, shuffle=False, dist=False, seed=0, 
        drop_last=False, samples_per_gpu=SPG, workers_per_gpu=WKR)
    data_loader = build_dataloader(dataset, **loader_cfg, ) 
    losses = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            img = batch['img'].data[0].cuda()
            # print(f"img {img.shape}")
            ground_truth = batch['gt_semantic_seg'].data[0].cuda()
            # print(f"ground_truth: {ground_truth.shape}")
            img_metas = batch['img_metas']
            curr_loss = network.forward_train(img, img_metas, ground_truth, stage='mae')
            np_curr_loss = curr_loss['decode.loss_rec'].cpu().numpy()
            losses.append(np_curr_loss)

    losses = np.array(losses)
    indices = losses.argsort() # ascending order
    if return_loss:
        return indices, losses
    else:
        return indices
    
def unnormalize(x, img_norm_cfg):
    mean, std = torch.Tensor(img_norm_cfg['mean']), torch.Tensor(img_norm_cfg['std'])
    mean, std = mean[:, None, None], std[:, None, None]
    return torch.clip((x.cpu() * std + mean).type(torch.uint8), 0, 255)

def get_entropy(p, dim, keepdim=False):
    return torch.sum(-p * torch.log(p + 1e-6), dim=dim, keepdim=keepdim)

def get_cross_entropy(p, q, dim, keepdim=False):
    return torch.sum(-p * torch.log(q + 1e-6), dim=dim, keepdim=keepdim)

def pixel_mean_entropy(softmax_pred, use_ce=False, q=None):
    bs, classes, img_H, img_W = softmax_pred.size()
    if use_ce:
        entropy_map = get_cross_entropy(softmax_pred, q, dim=1, keepdim=False)
    else:
        entropy_map = get_entropy(softmax_pred, dim=1, keepdim=False)
    entropy_map = entropy_map.reshape(shape=(bs, img_H, img_W))
    return entropy_map.cpu().numpy()