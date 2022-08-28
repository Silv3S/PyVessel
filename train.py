import wandb
import torch
import config
from tqdm import tqdm
from cmath import inf
from utils import save_model
from visualize import save_loss_history


def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler):
    train_loss = 0
    val_loss = 0
    train_loop = tqdm(train_loader)
    val_loop = tqdm(val_loader)

    model.train()
    for _, (x, y) in enumerate(train_loop):
        x = x.to(device=config.DEVICE)
        y = y.float().unsqueeze(1).to(device=config.DEVICE)
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.cpu().detach().numpy()

    # Validation, turn off for faster training
    if config.USE_VALIDATION_SET:
        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(val_loop):
                x = x.to(device=config.DEVICE)
                y = y.float().unsqueeze(1).to(device=config.DEVICE)
                predictions = model(x)
                loss = loss_fn(predictions, y)
                val_loss += loss.cpu().detach().numpy()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(
        f"Train loss: {round(avg_train_loss,4)}\nVal loss: {round(avg_val_loss,4)}")

    return avg_train_loss, avg_val_loss


class LossTracker():
    def __init__(self):
        self.tolerance = config.EARLY_STOP_PATIENCE
        self.min_delta = config.EARLY_STOP_DELTA
        self.counter = 0
        self.early_stop = False
        self.least_val_loss = inf
        self.train_stats = {'train_loss': [], 'val_loss': []}

    def __call__(self, model, training_loss, validation_loss):
        self.train_stats['train_loss'].append(training_loss)
        self.train_stats['val_loss'].append(validation_loss)

        if(config.SYNC_WANDB):
            wandb.log({"train_loss": training_loss,
                       "val_loss": validation_loss})

        if (validation_loss - self.least_val_loss) > self.min_delta:
            self.counter += 1
            print(
                f'Slowly losing patience ... ({self.counter}/{self.tolerance})\n')
            if self.counter >= self.tolerance:
                print("Validation loss is no longer decreasing.")
                print("Stop training to avoid overfitting.")
                self.early_stop = True
        else:
            self.counter = 0
            self.least_val_loss = validation_loss
            save_model(model.state_dict())

    def save_loss_plots(self):
        save_loss_history(self.train_stats)
