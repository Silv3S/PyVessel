import torch
from architectures.DAREUNet import DARE_UNet
import loss

# Data preparation
PREPARE_DATASETS = True
PATCH_SIZE = 128
PATCH_STEP_TRAIN = 64  # Used for data augmentation

# Hyperparameters
RANDOM_SEED = 1337
TEST_SET_RATIO = 0.15
VAL_SET_RATIO = 0.15
LR = 0.001
NUM_EPOCHS = 300
BATCH_SIZE = 16
MODEL_ARCHITECTURE = DARE_UNet()
LOSS_FUNC = loss.TverskyLoss(0.4, 0.6)
EARLY_STOP_PATIENCE = 13
EARLY_STOP_DELTA = 0

# Other
PROJECT_NAME = "Thesis DARE U-Net"
TRAIN_LIMITS = 0  # Run training on limited number of images
LOAD_PRETRAINED_MODEL = True
SAVE_TEST_RESULTS = True
SYNC_WANDB = True

# Visualization
TP_RGB = (0, 0, 0)
TN_RGB = (1, 1, 1)
FP_RGB = (19, 160, 191)
FN_RGB = (1, 0, 0)

# Filepaths
DATASETS_PATH = 'Datasets/'
TRAIN_DATASETS_PATH = 'Datasets_Train/'
TEST_DATASETS_PATH = 'Datasets_Test/'
PATCHES_PATH = './Patches/'
PLOTS_PATH = './Plots/'
BEST_MODEL_PATH = './Trained_models/PRETRAINED_DARE_UNet.pth'

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 1
