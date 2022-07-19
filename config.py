import torch
from architectures.UNet import UNet
import loss

patch_size = 256

# Data preparation
PREPARE_DATA = False
PATCH_SHAPE_GRAYSCALE = (patch_size, patch_size)
PATCH_SHAPE_RGB = (patch_size, patch_size, 3)
PATCH_STEP = 256

# Hyperparameters
RANDOM_SEED = 37
TEST_SPLIT = 0.15
VAL_SPLIT = 0.1
LR = 0.001
NUM_EPOCHS = 3
BATCH_SIZE = 4      # 8 is max for full precision fp32 GTX 1060 6GB
MODEL_ARCHITECTURE = UNet()
LOSS_FUNC = loss.DiceBCELoss()

# Other
USE_VALIDATION_SET = True
LOAD_PRETRAINED_MODEL = False
SAVE_TRAINING_RESULTS = True
SAVE_TEST_RESULTS = True

# Filepaths
DATASETS_PATH = './Datasets/'
IMAGES_LIST = './images_list.txt'
PLOTS_PATH = './Plots/'
PATCHED_IMAGES_PATH = './Patches/src/'
PATCHED_MASKS_PATH = './Patches/mask/'
TEST_IMAGES_PATH = './test_images_list.txt'
TEST_MASKS_PATH = './test_masks_list.txt'
BEST_MODEL_PATH = './Trained_models/UNet.pth'
LOSS_PLOT_PATH = './Plots/train_val_loss.png'

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 1
