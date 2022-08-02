import time
import torch
import config
from data_preparation import extract_train_patches, split_train_test_images
from dataset import get_train_dataloaders
from train import EarlyStopping, train_fn
from utils import clear_image_directories, load_model
from torch.optim import Adam
from visualize import plot_loss_history
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
    train_stats = {'train_loss': [], 'val_loss': []}

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    early_stopping = EarlyStopping(tolerance=5, min_delta=0.01)
    start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch: {epoch + 1} of {config.NUM_EPOCHS}')
        train_loss, val_loss = train_fn(
            train_loader, val_loader, model, opt, loss_func, scaler)

        train_stats['train_loss'].append(train_loss)
        train_stats['val_loss'].append(val_loss)

        early_stopping(model, val_loss)
        if(early_stopping.early_stop):
            print("Validation loss is no longer decreasing. Stop to avoid overfitting")
            break

    print('That\'s all Folks!')
    print(f'Total training time: {(time.time() - start_time):.2f}s')
    plot_loss_history(train_stats)
