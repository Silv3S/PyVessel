import time
import torch
import config
from dataset import get_dataloader, get_train_dataloaders
from test import evaluate_model
from train import LossTracker, train_fn
from utils import list_directory, load_model, parse_cli_args
import wandb


if __name__ == '__main__':
    parse_cli_args()

    if(config.SYNC_WANDB):
        wandb.init(project=config.PROJECT_NAME)
        wandb.config = {
            "learning_rate": config.LR,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "patch_size": config.PATCH_SIZE,
            "test_set_ratio": config.TEST_SET_RATIO,
            "val_set_ratio": config.VAL_SET_RATIO,
            "random_seed": config.RANDOM_SEED,
        }

    model = config.MODEL_ARCHITECTURE.to(config.DEVICE)
    if(config.LOAD_PRETRAINED_MODEL):
        load_model(torch.load(config.BEST_MODEL_PATH), model)

    loss_func = config.LOSS_FUNC
    opt = torch.optim.Adam(model.parameters(), config.LR)
    scaler = torch.cuda.amp.GradScaler()

    loss_tracker = LossTracker()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    (train_loader, val_loader) = get_train_dataloaders()

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

    test_image_paths, test_mask_paths = list_directory(
        config.TEST_DATASETS_PATH)
    test_loader = get_dataloader(test_image_paths, test_mask_paths, False, 1)

    model.eval()
    evaluate_model(model, test_loader)
