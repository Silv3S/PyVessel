import torch
from architectures.UNet import UNet
import loss

patch_size = 256

# Data preparation
PREPARE_DATASETS = False
PATCH_SHAPE_MASK = (patch_size, patch_size)
PATCH_SHAPE_IMG = (patch_size, patch_size, 3)
PATCH_SHAPE_IMG_NHWC = (1, 3, patch_size, patch_size)
PATCH_STEP = 256

# Hyperparameters
RANDOM_SEED = 37
TEST_SET_RATIO = 0.15
VAL_SET_RATIO = 0.1
LR = 0.001
NUM_EPOCHS = 3
BATCH_SIZE = 4      # 8 is max for full precision fp32 GTX 1060 6GB
MODEL_ARCHITECTURE = UNet()
LOSS_FUNC = loss.DiceBCELoss()

# Other
USE_VALIDATION_SET = True
LOAD_PRETRAINED_MODEL = False
SAVE_TEST_RESULTS = True

# Filepaths
DATASETS_PATH = 'Datasets/'
TRAIN_DATASETS_PATH = 'Datasets_Train/'
TEST_DATASETS_PATH = 'Datasets_Test/'
PATCHES_PATH = './Patches/'
PLOTS_PATH = './Plots/'
BEST_MODEL_PATH = './Trained_models/UNet.pth'
LOSS_PLOT_PATH = './Plots/train_val_loss.png'

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 1
