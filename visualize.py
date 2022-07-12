from matplotlib.ticker import MaxNLocator
import config
from matplotlib import pyplot as plt


def plot_loss_history(stats):
    plt.figure()
    ax = plt.gca()
    plt.plot(stats["train_loss"], label="train_loss")
    plt.plot(stats["validation_loss"], label="validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper right")
    plt.style.use("ggplot")
    plt.savefig(config.LOSS_PLOT_PATH)
