import shutil
import os
import numpy as np
import config
import random
from patchify import patchify
from skimage.io import imread, imsave


def split_train_test_images():
    random.seed(config.RANDOM_SEED)
    for dataset in os.listdir(config.DATASETS_PATH):
        images_count = len(os.listdir(config.DATASETS_PATH + dataset + '/src'))
        for i in range(1, images_count+1):
            target_directory = config.TEST_DATASETS_PATH if random.random(
            ) <= config.TEST_SET_RATIO else config.TRAIN_DATASETS_PATH
            shutil.copyfile(config.DATASETS_PATH + dataset + '/src/' + dataset + str(i) + '.png',
                            target_directory + '/src/' + dataset + str(i) + '.png')
            shutil.copyfile(config.DATASETS_PATH + dataset + '/mask/' + dataset + str(i) + '_mask.png',
                            target_directory + '/mask/' + dataset + str(i) + '.png')
        print(
            f'Loaded dataset {dataset} with {images_count} images and masks.')

    train_set_size = len(os.listdir(config.TRAIN_DATASETS_PATH + "src/"))
    test_set_size = len(os.listdir(config.TEST_DATASETS_PATH + "src/"))
    print(
        f"Train/Val set: {train_set_size} images\nTest set: {test_set_size} images")


def extract_train_patches():
    for path in os.listdir(config.TRAIN_DATASETS_PATH + "src/"):
        image = imread(config.TRAIN_DATASETS_PATH + "src/" + path)
        mask = imread(config.TRAIN_DATASETS_PATH + "mask/" + path)
        image, mask = add_zero_padding(image, mask)
        image_patches = patchify(
            image, config.PATCH_SHAPE_IMG, step=config.PATCH_STEP_TRAIN)
        mask_patches = patchify(
            mask, config.PATCH_SHAPE_MASK, step=config.PATCH_STEP_TRAIN)
        save_extracted_patches(
            image_patches, mask_patches, path[:-4])
    patches_count = len(os.listdir(config.PATCHES_PATH + "src/"))
    print(f"Divided training set to {patches_count} patches")


def save_extracted_patches(image_patches, mask_patches, image_name):
    idx = 0
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            if(is_patch_useless(mask_patches[i, j, ...])):
                continue

            image_patch = image_patches[i, j, ...]
            if(len(image_patch.shape) == 4):
                image_patch = np.squeeze(image_patch)

            filename = f'{image_name}_patch{idx}.png'
            imsave(config.PATCHES_PATH + 'src/' + filename,
                   image_patch, check_contrast=False)
            imsave(config.PATCHES_PATH + 'mask/' + filename,
                   mask_patches[i, j, ...], check_contrast=False)
            idx = idx + 1


def is_patch_useless(mask_patch):
    """
    Patch is considered useless if the blood vessel pixels are less than 1% of the patch
    """
    pixel_count = np.prod(mask_patch.shape)
    blood_pixel_count = np.count_nonzero(mask_patch == 255)
    return blood_pixel_count/pixel_count < 0.01


def add_zero_padding(image, mask, format_NHWC=False):
    """
    Patchify can't handle under-sized patches. Without padding dataset is not fully utilised
    """
    w = 2 if format_NHWC else 0
    h = 3 if format_NHWC else 1
    h_pad = config.PATCH_SHAPE_MASK[0] - \
        (image.shape[w] % config.PATCH_SHAPE_MASK[0])
    v_pad = config.PATCH_SHAPE_MASK[1] - \
        (image.shape[h] % config.PATCH_SHAPE_MASK[1])

    if(h_pad != 0 or v_pad != 0):
        if(format_NHWC):
            img_pad = [(0, 0), (0, 0), (0, h_pad), (0, v_pad)]
            mask_pad = [(0, 0), (0, h_pad), (0, v_pad)]
        else:
            img_pad = [(0, h_pad), (0, v_pad), (0, 0)]
            mask_pad = [(0, h_pad), (0, v_pad)]
        image = np.pad(image, img_pad, mode='constant', constant_values=0)
        mask = np.pad(mask, mask_pad, mode='constant', constant_values=0)
    return image, mask
