import uuid
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import config
from torchvision.utils import save_image


def save_loss_history(stats):
    plt.figure()
    ax = plt.gca()
    plt.plot(stats["train_loss"], label="train_loss")
    plt.plot(stats["val_loss"], label="validation_loss")
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
