import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, Resize, InterpolationMode as IM
from plotting.utils import unnormalize

crop_size = (384, 384)
tensor2image = ToPILImage(mode='RGB')
tensor_resize = Resize(size=crop_size, interpolation=IM.NEAREST)


def vis_original_with_mae(original, mae_pred, img_norm_cfg):
    rows, columns = 1, 2
    rgb_prediction = unnormalize(mae_pred.squeeze(0).detach(), img_norm_cfg)
    pil_prediction = tensor2image(rgb_prediction)
    # define plot figure
    fig = plt.figure(figsize=(15, 8))
    # first
    fig.add_subplot(rows, columns, 1)
    plt.title("original image")
    plt.imshow(tensor2image(unnormalize(original, img_norm_cfg)))
    plt.axis('off')
    # second
    fig.add_subplot(rows, columns, 2)
    plt.title("reconstruction")
    plt.imshow(pil_prediction)
    plt.axis('off')

def vis_uncertainties_and_losses(
    entropy_score, ce_score, mae_score, ripu_score):
    rows, columns = 2, 2
    fig = plt.figure(figsize=(14, 12))
    fig.add_subplot(rows, columns, 1)
    # first
    plt.title("prediction entropy (pixel)")
    plt.imshow(entropy_score, cmap='viridis', alpha=.8, )
    plt.axis('off')
    plt.colorbar()
    # second
    fig.add_subplot(rows, columns, 2)
    plt.title("reconstruction loss")
    plt.imshow(mae_score, cmap='viridis', alpha=.8)
    plt.axis('off')
    plt.colorbar()
    # third
    fig.add_subplot(rows, columns, 3)
    plt.title("cross entropy loss (pixel)")
    plt.imshow(ce_score, cmap='viridis', alpha=.8, )
    plt.axis('off')
    plt.colorbar()
    # fourth
    fig.add_subplot(rows, columns, 4)
    plt.title("RIPU score")
    plt.imshow(ripu_score, cmap='viridis', alpha=.8)
    plt.axis('off')
    plt.colorbar()