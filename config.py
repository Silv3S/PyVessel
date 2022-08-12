import torch
from architectures.LadderNet import LadderNet
from architectures.UNet import UNet
import loss

# Data preparation
PREPARE_DATASETS = False
PATCH_SIZE = 256
PATCH_STEP_TRAIN = 128  # Used for data augmentation

# Hyperparameters
RANDOM_SEED = 37
TEST_SET_RATIO = 0.15
VAL_SET_RATIO = 0.1
LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 8      # 8 is max for full precision fp32 GTX 1060 6GB
MODEL_ARCHITECTURE = LadderNet()
LOSS_FUNC = loss.TverskyLoss(0.3, 0.7)
EARLY_STOP_PATIENCE = 5
EARLY_STOP_DELTA = 0.1

# Other
PROJECT_NAME = "PyVessel"
TRAIN_LIMITS = 0  # Run training on limited number of images
USE_VALIDATION_SET = True
LOAD_PRETRAINED_MODEL = False
SAVE_TEST_RESULTS = False
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
BEST_MODEL_PATH = './Trained_models/UNet.pth'
LOSS_PLOT_PATH = './Plots/train_val_loss.png'

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
NUM_WORKERS = 1
