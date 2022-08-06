from imutils import paths
from glob import glob
import os
import torch
import config
from data_preparation import extract_train_patches, split_train_test_images


def prepare_datasets():
    clear_image_directories()
    split_train_test_images()
    extract_train_patches()


def clear_image_directories():
    directories = [config.TRAIN_DATASETS_PATH + "src/*", config.TRAIN_DATASETS_PATH + "mask/*",
                   config.TEST_DATASETS_PATH + "src/*", config.TEST_DATASETS_PATH + "mask/*",
                   config.PATCHES_PATH + "src/*", config.PATCHES_PATH + "mask/*"]
    for dir in directories:
        for file in glob(dir):
            os.remove(file)


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


def average(list, round_to):
    return round(sum(list) / len(list), round_to)


def list_directory(directory):
    image_paths = sorted(list(paths.list_images(directory + "src/")))
    mask_paths = sorted(list(paths.list_images(directory + "mask/")))
    return image_paths, mask_paths
