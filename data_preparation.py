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
                            target_directory + '/mask/' + dataset + str(i) + '_mask.png')
        print(
            f'Loaded dataset {dataset} with {images_count} images and masks.')

    train_set_size = len(os.listdir(config.TRAIN_DATASETS_PATH + "src/"))
    test_set_size = len(os.listdir(config.TEST_DATASETS_PATH + "src/"))
    print(f"Train/Val set: {train_set_size}\nTest set: {test_set_size}")


def extract_train_patches():
    for path in os.listdir(config.TRAIN_DATASETS_PATH + "src/"):
        image = imread(config.TRAIN_DATASETS_PATH + "src/" + path)
        mask = imread(config.TRAIN_DATASETS_PATH +
                      "mask/" + path[:-4] + "_mask.png")

        patch_shape = config.PATCH_SHAPE_GRAYSCALE if len(
            image.shape) == 2 else config.PATCH_SHAPE_RGB
        image_patches = patchify(image, patch_shape, step=config.PATCH_STEP)
        mask_patches = patchify(
            mask, config.PATCH_SHAPE_GRAYSCALE, step=config.PATCH_STEP)

        save_extracted_patches(
            image_patches, mask_patches, path[:-4])


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
            imsave('./Patches/src/' + filename,
                   image_patch, check_contrast=False)
            imsave('./Patches/mask/' + filename,
                   mask_patches[i, j, ...], check_contrast=False)
            idx = idx + 1
    patches_count = len(os.listdir(config.PATCHES_PATH + "src/"))
    print(f"Divided training set to {patches_count} patches")


def is_patch_useless(mask_patch):
    """ Patch is considered useless if the blood vessel pixels are less than 1% of the patch """
    pixel_count = np.prod(mask_patch.shape)
    blood_pixel_count = np.count_nonzero(mask_patch == 255)
    return blood_pixel_count/pixel_count < 0.01
