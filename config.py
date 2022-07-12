import torch
from architectures.UNet import UNet
from torch.nn import BCEWithLogitsLoss


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
BATCH_SIZE = 4      # 32 is max for GTX 1060 6GB
MODEL_ARCHITECTURE = UNet((patch_size, patch_size))
LOSS_FUNC = BCEWithLogitsLoss()

# Other
USE_VALIDATION_SET = True

# Filepaths
DATASETS_PATH = './Datasets/'
IMAGES_LIST = './images_list.txt'
PATCHED_IMAGES_PATH = './MVPatches/src/'
PATCHED_MASKS_PATH = './MVPatches/mask/'
TEST_IMAGES_PATH = './test_images_list.txt'
TEST_MASKS_PATH = './test_masks_list.txt'
BEST_MODEL_PATH = './Trained_models/UNet.pt'
LOSS_PLOT_PATH = './Plots/train_val_loss.png'

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
