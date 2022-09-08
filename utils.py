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
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', dest='epochs',
                        type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=config.LR)
    parser.add_argument('--patch_size', dest='patch_size',
                        type=int, default=config.PATCH_SIZE)
    parser.add_argument('--random_seed', dest='random_seed',
                        type=int, default=config.RANDOM_SEED)
    parser.add_argument('--val_set_ratio', dest='val_set_ratio',
                        type=float, default=config.VAL_SET_RATIO)
    parser.add_argument('--project_name', dest='project_name',
                        default=config.PROJECT_NAME)
    parser.add_argument('--prepare_new_dataset',
                        dest='prepare_new_dataset', type=bool, default=False)
    args = parser.parse_args()

    config.SYNC_WANDB = True
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LR = args.learning_rate
    config.PATCH_SIZE = args.patch_size
    config.RANDOM_SEED = args.random_seed
    config.VAL_SET_RATIO = args.val_set_ratio
    config.PROJECT_NAME = args.project_name
