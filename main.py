import time
import torch
import config
from dataset import get_train_dataloaders
from train import LossTracker, train_fn
from utils import prepare_datasets, load_model
import wandb


if __name__ == '__main__':
    wandb.init(project="PyVessel")
    wandb.config = {
        "learning_rate": config.LR,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "patch_size": config.PATCH_SHAPE_IMG,
        "test_set_ratio": config.TEST_SET_RATIO,
        "val_set_ratio": config.VAL_SET_RATIO,
        "random_seed": config.RANDOM_SEED,
    }

    if(config.PREPARE_DATASETS):
        prepare_datasets()

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
        wandb.log({"train_loss": train_loss})
        wandb.log({"val_loss": val_loss})

    print('That\'s all Folks!')
    print(f'Total training time: {(time.time() - start_time):.2f}s')
    loss_tracker.save_loss_plots()
