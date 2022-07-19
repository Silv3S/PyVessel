import uuid
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import config
from torchvision.utils import save_image


def plot_loss_history(stats):
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


def plot_results_inline(org_img, org_mask, seg_result):
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
        filename = f'{config.PLOTS_PATH}_{uuid.uuid4()}.png'
        plt.savefig(filename)

    plt.close(fig)


def visualize_training_results(loader, model):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=config.DEVICE)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        save_image(preds, f"{config.PLOTS_PATH}/pred_{idx}.png")
        save_image(y.unsqueeze(1), f"{config.PLOTS_PATH}{idx}.png")

    model.train()
