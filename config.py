import torch


patch_size = 256

PREPARE_DATA = False
DATASETS_PATH = './Datasets/'
IMAGES_LIST = './images_list.txt'
PATCHED_IMAGES_PATH = './Patches/src/'
PATCHED_MASKS_PATH = './Patches/mask/'
PATCH_SHAPE_GRAYSCALE = (patch_size, patch_size)
PATCH_SHAPE_RGB = (patch_size, patch_size, 3)
PATCH_STEP = 256

RANDOM_SEED = 37
TEST_SPLIT = 0.15
