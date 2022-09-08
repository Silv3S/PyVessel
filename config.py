import torch
from architectures.DAREUNet import DARE_UNet
import loss

dataset_name = 'drive'

# Data preparation
PATCH_SIZE = 128

# Hyperparameters
RANDOM_SEED = 1337
VAL_SET_RATIO = 0.15
LR = 0.001
NUM_EPOCHS = 300
BATCH_SIZE = 16
MODEL_ARCHITECTURE = DARE_UNet()
LOSS_FUNC = loss.DiceLoss()
EARLY_STOP_PATIENCE = 6
EARLY_STOP_DELTA = 0

# Other
PROJECT_NAME = "Thesis DARE U-Net"
PRETRAINED_MODEL_PATH = './Trained_models/PRETRAINED_DARE_UNet.pth'
SAVE_TEST_RESULTS = True
SYNC_WANDB = True

# Visualization
TP_RGB = (0, 0, 0)
TN_RGB = (1, 1, 1)
FP_RGB = (19, 160, 191)
FN_RGB = (1, 0, 0)

# Filepaths
TEST_DATASETS_PATH = 'Datasets_Test_One/' + dataset_name + '/'
PATCHES_PATH = './Patches_One/' + dataset_name + '/'
PLOTS_PATH = './Plots_One/'
BEST_MODEL_PATH = './Trained_models/' + dataset_name + '.pth'

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 1
