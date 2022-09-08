import argparse
from imutils import paths
import torch
import config


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
    parser.add_argument('--dataset_name', dest='dataset_name')
    args = parser.parse_args()

    config.SYNC_WANDB = True
    config.dataset_name = args.dataset_name
    config.TEST_DATASETS_PATH = 'Datasets_Test_One/' + args.dataset_name + '/'
    config.PATCHES_PATH = './Patches_One/' + args.dataset_name + '/'
    config.PLOTS_PATH = './Plots_One/'
    config.BEST_MODEL_PATH = './Trained_models/' + args.dataset_name + '.pth'
