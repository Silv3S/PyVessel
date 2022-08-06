import time
import torch
import config
from data_preparation import extract_train_patches, split_train_test_images
from dataset import get_train_dataloaders
from train import LossTracker, train_fn
from utils import clear_image_directories, load_model
from torch.optim import Adam


if __name__ == '__main__':
    if(config.PREPARE_DATASETS):
        clear_image_directories()
        split_train_test_images()
        extract_train_patches()

    (train_loader, val_loader) = get_train_dataloaders()

    device = config.DEVICE
    model = config.MODEL_ARCHITECTURE.to(device)
    loss_func = config.LOSS_FUNC
    opt = Adam(model.parameters(), config.LR)

    if config.LOAD_PRETRAINED_MODEL:
        load_model(torch.load(config.BEST_MODEL_PATH), model)

    scaler = torch.cuda.amp.GradScaler()

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    loss_tracker = LossTracker()
    start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch: {epoch + 1} of {config.NUM_EPOCHS}')
        train_loss, val_loss = train_fn(
            train_loader, val_loader, model, opt, loss_func, scaler)

        loss_tracker(model, train_loss, val_loss)
        if(loss_tracker.early_stop):
            break

    print('That\'s all Folks!')
    print(f'Total training time: {(time.time() - start_time):.2f}s')
    loss_tracker.save_loss_plots()
