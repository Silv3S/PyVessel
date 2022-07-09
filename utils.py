from matplotlib.pyplot import get
import numpy as np
from requests import patch
from skimage.io import imread, imsave
import os
from patchify import patchify
from pathlib import Path
from config import Config


def create_images_list():
    datasets_path = Config.Get("datasets_path")
    save_path = Config.Get("images_list")
    if os.path.exists(save_path):
        os.remove(save_path)
        print("Old image list removed")

    for dataset in os.listdir(datasets_path):
        images_count = len(os.listdir(datasets_path + dataset + '/src'))
        f = open(save_path, 'a')
        for i in range(1, images_count+1):
            f.write(datasets_path + dataset + '/src/' + dataset + str(i) + '.png ' +
                    datasets_path + dataset + '/mask/' + dataset + str(i) + '_mask.png' + '\n')
        print(
            f'Loaded dataset {dataset} with {images_count} images and masks.')


def extract_patches():
    img_list = open(Config.Get('images_list'), 'r')
    image_pairs = img_list.readlines()

    for image_pair in image_pairs:
        image_pair = image_pair.strip().split()
        image = imread(image_pair[0])
        mask = imread(image_pair[1])

        patch_shape = Config.Get('patch_shape_grayscale') if len(
            image.shape) == 2 else Config.Get('patch_shape_rgb')
        patch_step = Config.Get('patch_step')
        image_patches = patchify(image, patch_shape, step=patch_step)
        mask_patches = patchify(mask, Config.Get(
            'patch_shape_grayscale'), step=patch_step)

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
