import uuid
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import config
from torchvision.utils import save_image
from skimage.io import imread, imsave
from patchify import patchify
from data_preparation import add_zero_padding


def save_loss_history(stats):
    plt.figure()
    ax = plt.gca()
    x_count = len(stats["train_loss"])
    plt.plot(range(1, x_count + 1), stats["train_loss"], label="train_loss")
    plt.plot(range(1, x_count + 1), stats["val_loss"], label="validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    ax.set_ylim(ymin=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper right")
    plt.style.use("ggplot")
    plt.savefig(config.LOSS_PLOT_PATH)


def plot_results_inline(org_img, org_mask, seg_result, img_id=uuid.uuid4()):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax[0, 0].imshow(org_img)
    ax[0, 0].set_title("Original image")
    ax[0, 1].imshow(org_mask)
    ax[0, 1].set_title("Ground truth")
    ax[1, 0].imshow(org_img)
    ax[1, 0].imshow(seg_result,  alpha=0.7)
    ax[1, 0].set_title("Segmentation result")
    ax[1, 1].imshow(seg_result)
    ax[1, 1].set_title("Segmentation result")
    fig.tight_layout()
    for axis in ax.flat:
        axis.set_axis_off()
    if config.SAVE_TEST_RESULTS:
        filename = f'{config.PLOTS_PATH}_{img_id}.png'
        plt.savefig(filename)

    plt.close(fig)


def visualize_training_results(loader, model):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=config.DEVICE)

        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()
        save_image(preds, f"{config.PLOTS_PATH}/pred_{idx}.png")
        save_image(y.unsqueeze(1), f"{config.PLOTS_PATH}{idx}.png")

    model.train()


def save_graphical_confusion_matrix(y_true, y_pred, img, img_id=uuid.uuid4()):
    y_pred = y_pred / 255.0
    tp = (y_true+y_pred) == 2
    tn = (y_true+y_pred) == 0
    fp = (y_true-y_pred) == -1
    fn = (y_true-y_pred) == 1

    cm = np.zeros((y_true.shape[0], y_true.shape[1], 3))
    for i in range(0, cm.shape[0]):
        for j in range(0, cm.shape[1]):
            if(tn[i, j] == 1):
                cm[i, j, :] = config.TP_RGB
            elif(tp[i, j] == 1):
                cm[i, j, :] = config.TN_RGB
            elif(fp[i, j] == 1):
                cm[i, j, :] = config.FP_RGB
                cm[i, j, :] /= 255
            elif(fn[i, j] == 1):
                cm[i, j, :] = config.FN_RGB

    matplotlib.image.imsave(
        f'{config.PLOTS_PATH}_CM_{img_id}.png', np.hstack((img, cm)))


def save_patched_image(image, mask, padding):
    image, mask = add_zero_padding(image, mask)
    image_patches = patchify(
        image, config.PATCH_SHAPE_IMG, step=config.PATCH_STEP)
    mask_patches = patchify(
        mask, config.PATCH_SHAPE_MASK, step=config.PATCH_STEP)
    x_patch_count = image_patches.shape[0]
    y_patch_count = image_patches.shape[1]
    x_dim = x_patch_count * config.patch_size + (x_patch_count - 1) * padding
    y_dim = y_patch_count * config.patch_size + (y_patch_count - 1) * padding
    patched_padded_image = np.ones((x_dim, y_dim, 3)) * 255.0
    patched_padded_mask = np.ones((x_dim, y_dim)) * 255.0

    x_start = [(config.patch_size + padding) * x
               for x in range(0, x_patch_count+1)]
    x_end = [x + config.patch_size for x in x_start]
    y_start = [(config.patch_size + padding) * y
               for y in range(0, y_patch_count+1)]
    y_end = [y + config.patch_size for y in y_start]

    for i in range(x_patch_count):
        for j in range(y_patch_count):
            patched_padded_image[x_start[i]:x_end[i],
                                 y_start[j]:y_end[j], :] = image_patches[i, j]
            patched_padded_mask[x_start[i]:x_end[i],
                                y_start[j]:y_end[j]] = mask_patches[i, j]

    imsave(f"{config.PLOTS_PATH}/patched_img.png",
           patched_padded_image, check_contrast=False)
    imsave(f"{config.PLOTS_PATH}/patched_mask.png",
           patched_padded_mask, check_contrast=False)


if __name__ == '__main__':
    image = imread("./Datasets/HRF/src/HRF3.png")
    mask = imread("./Datasets/HRF/mask/HRF3_mask.png")
    save_patched_image(image, mask, 7)
