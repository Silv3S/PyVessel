import numpy as np
from skimage.io import imread, imsave
import os
from patchify import patchify
from pathlib import Path
import torch
import config


def create_images_list():
    if os.path.exists(config.IMAGES_LIST_PATH):
        os.remove(config.IMAGES_LIST_PATH)
        print("Old image list removed")

    for dataset in os.listdir(config.DATASETS_PATH):
        images_count = len(os.listdir(config.DATASETS_PATH + dataset + '/src'))
        f = open(config.IMAGES_LIST_PATH, 'a')
        for i in range(1, images_count+1):
            f.write(config.DATASETS_PATH + dataset + '/src/' + dataset + str(i) + '.png ' +
                    config.DATASETS_PATH + dataset + '/mask/' + dataset + str(i) + '_mask.png' + '\n')
            f.close()
        print(
            f'Loaded dataset {dataset} with {images_count} images and masks.')


def extract_patches():
    image_pairs = config.IMAGES_LIST_PATH.readlines()

    for image_pair in image_pairs:
        image_pair = image_pair.strip().split()
        image = imread(image_pair[0])
        mask = imread(image_pair[1])

        patch_shape = config.PATCH_SHAPE_GRAYSCALE if len(
            image.shape) == 2 else config.PATCH_SHAPE_RGB
        image_patches = patchify(image, patch_shape, step=config.PATCH_STEP)
        mask_patches = patchify(
            mask, config.PATCH_SHAPE_GRAYSCALE, step=config.PATCH_STEP)

        save_extracted_patches(
            image_patches, mask_patches, Path(image_pair[0]).stem)
        print(image_pair)


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


def is_patch_useless(mask_patch):
    pixel_count = np.prod(mask_patch.shape)
    blood_pixel_count = np.count_nonzero(mask_patch == 255)
    return blood_pixel_count/pixel_count < 0.01


def save_to_file(filepath, content):
    f = open(filepath, "w")
    f.write("\n".join(content))
    f.close()


def read_txt_as_list(filepath):
    file = open(filepath, "r")
    content = file.read()
    file.close()
    return content.split("\n")


def save_model(checkpoint):
    torch.save(checkpoint, config.BEST_MODEL_PATH)


def load_model(checkpoint, model):
    model.load_state_dict(checkpoint)


def average(list):
    return round(sum(list) / len(list), 4)
