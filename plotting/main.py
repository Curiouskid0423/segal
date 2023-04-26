import numpy as np
import torch
from torchvision.transforms import ToPILImage, Resize, InterpolationMode as IM
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from segal.utils.heuristics import to_prob
from plotting.utils import onehot_with_ignore_label, pixel_mean_entropy
from plotting.visualization import vis_uncertainties_and_losses, vis_original_with_mae

crop_size = (384, 384)
tensor2image = ToPILImage(mode='RGB')
tensor_resize = Resize(size=crop_size, interpolation=IM.NEAREST)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

from segal.active.heuristics import RegionImpurity
ripu_engine = RegionImpurity(mode='pixel', categories=19, k=1)

def engine(
        dataset, idx, seg_net, mae_net, 
        mask_ratio=None, plot=False, truncation=0.1):

    BATCH = dataset[idx]
    IMG = BATCH['img'].data.cuda()
    GT = BATCH['gt_semantic_seg'].data.cuda()

    if mask_ratio != None:
        assert isinstance(mask_ratio, float) and 0. <= mask_ratio < 1. 
        mae_net.mask_ratio = mask_ratio

    with torch.no_grad():

        # mask prediction
        mae_net = mae_net.cuda()
        mae_prediction, mask = mae_net.mae_inference(IMG, return_mask=True)
        mae_prediction = mae_prediction.cpu().squeeze(0)
        mask = mask.cpu().squeeze(0)
        
        # segmentation entropy
        seg_net = seg_net.cuda()
        logits = seg_net.whole_inference(IMG.unsqueeze(0), None, rescale=False) # [1, 19, 640, 1280]
        
        # mask out ignore_label in ground_truth
        one_hot_gt = onehot_with_ignore_label(GT, num_classes=19, ignore_label=255)
        softmax_pred = to_prob(logits).cpu().to(float)
        
        pixel_cross_entropy = pixel_mean_entropy(
                softmax_pred, use_ce=True, q=one_hot_gt.cpu()).squeeze(0)
        pixel_entropy = pixel_mean_entropy(softmax_pred).squeeze(0)
        
        # compute ripu if inquired
        pixel_ripu = ripu_engine.get_uncertainties(logits).squeeze(0)

        # mae section
        unit = int(mask.shape[-1]**0.5)
        mask_2d = mask.detach().reshape(unit, unit).unsqueeze(0)
        scaled_mask_2d = tensor_resize(mask_2d).squeeze(0).to(bool)
        masked_mae_loss: torch.Tensor = (IMG.cpu() - mae_prediction)**2 * 0.5
        masked_mae_loss = masked_mae_loss.mean(dim=0)
        # masked_mae_loss = masked_mae_loss.sum(dim=0)

        # calculate the correlation coefficient 
        mae_rec_loss = masked_mae_loss[scaled_mask_2d].numpy()
        seg_pix_entropy = pixel_entropy[scaled_mask_2d]
        # define threshold (lower bound or upper bound)
        cap = int(len(seg_pix_entropy) * 0.01)
        if mask_ratio == 0.:
            # filter out the unimportant pixels for better visualization
            # truncate mae
            trunc_mae = masked_mae_loss.quantile(q=truncation)
            masked_mae_loss[masked_mae_loss < trunc_mae] = 0. 
            # truncate entropy
            trunc_entr = np.quantile(pixel_entropy, q=truncation)
            pixel_entropy[pixel_entropy < trunc_entr] = 0.
            # truncate cross entropy
            trunc_cross_entr = np.quantile(pixel_cross_entropy, q=truncation)
            pixel_cross_entropy[pixel_cross_entropy < trunc_cross_entr] = 0.
            # truncate ripu
            trunc_ripu = np.quantile(pixel_ripu, q=truncation)
            pixel_ripu[pixel_ripu < trunc_ripu] = 0.
            
            # visualize images
            vis_original_with_mae(IMG, mae_prediction, img_norm_cfg)
            # visualize two uncertainties
            # vis_entropy_and_mae_loss(pixel_entropy, masked_mae_loss)
            vis_uncertainties_and_losses(
                entropy_score = pixel_entropy, 
                ce_score = pixel_cross_entropy, 
                mae_score = masked_mae_loss, 
                ripu_score = pixel_ripu
            )
            return None
        reordered = np.argsort(mae_rec_loss)[-cap:]
        S, M = seg_pix_entropy[reordered], mae_rec_loss[reordered]
        corr, _ = pearsonr(S, M)
        
        if plot:
            print(f"there are {len(mae_rec_loss)} sample points. set threshold at {cap}.")
            # plot the losses
            plt.figure(figsize=(10, 6))
            plt.title('seg_pix_entropy vs mae_rec_loss')
            plt.scatter(S, M)
            plt.xlabel('seg_pix_entropy')
            plt.ylabel('mae_rec_loss')
            print('plotted coeff:  %.4f' % corr)

            # visualize images
            vis_original_with_mae(IMG, mae_prediction, img_norm_cfg)
            # visualize two uncertainties
            vis_entropy_and_mae_loss(
                pixel_entropy, 
                masked_mae_loss
            )
        return corr