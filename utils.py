from architectures.AttentionUNet import AttentionUNet
from architectures.LadderNet import LadderNet
from architectures.R2UNet import R2UNet
from architectures.SAUnet import SA_UNet
from architectures.UNet import UNet
import data_preparation
import argparse
from imutils import paths
from glob import glob
import os
import torch
import config
import sys


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


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', dest='epochs',
                        type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=config.LR)
    parser.add_argument('--patch_size', dest='patch_size',
                        type=int, default=config.PATCH_SIZE)
    parser.add_argument('--patch_step_train',
                        dest='patch_step_train', type=int, default=config.PATCH_STEP_TRAIN)
    parser.add_argument('--random_seed', dest='random_seed',
                        type=int, default=config.RANDOM_SEED)
    parser.add_argument('--val_set_ratio', dest='val_set_ratio',
                        type=float, default=config.VAL_SET_RATIO)
    parser.add_argument('--test_set_ratio', dest='test_set_ratio',
                        type=float, default=config.TEST_SET_RATIO)
    parser.add_argument('--limits', dest='limits', type=int,
                        default=config.TRAIN_LIMITS)
    parser.add_argument('--project_name', dest='project_name',
                        default=config.PROJECT_NAME)
    parser.add_argument('--model_name', dest='model_name', type=str,
                        default="")
    parser.add_argument('--prepare_new_dataset',
                        dest='prepare_new_dataset', type=bool, default=False)
    args = parser.parse_args()

    config.SYNC_WANDB = True
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LR = args.learning_rate
    config.PATCH_SIZE = args.patch_size
    config.PATCH_STEP_TRAIN = args.patch_step_train
    config.RANDOM_SEED = args.random_seed
    config.VAL_SET_RATIO = args.val_set_ratio
    config.TEST_SET_RATIO = args.test_set_ratio
    config.TRAIN_LIMITS = args.limits
    config.PROJECT_NAME = args.project_name

    if(args.model_name == "base_unet"):
        config.MODEL_ARCHITECTURE = UNet()
        config.BEST_MODEL_PATH = './Trained_models/UNet.pth'
    elif(args.model_name == "r2_unet"):
        config.MODEL_ARCHITECTURE = R2UNet()
        config.BEST_MODEL_PATH = './Trained_models/R2UNet.pth'
    elif(args.model_name == "attention_unet"):
        config.MODEL_ARCHITECTURE = AttentionUNet()
        config.BEST_MODEL_PATH = './Trained_models/AtNet.pth'
    elif(args.model_name == "ladder_net"):
        config.MODEL_ARCHITECTURE = LadderNet()
        config.BEST_MODEL_PATH = './Trained_models/LadderNet.pth'
    elif(args.model_name == "sa_unet"):
        config.MODEL_ARCHITECTURE = SA_UNet()
        config.BEST_MODEL_PATH = './Trained_models/SAUNet.pth'

    if(args.prepare_new_dataset):
        data_preparation.prepare_datasets()
        sys.exit("New dataset is loaded!")
