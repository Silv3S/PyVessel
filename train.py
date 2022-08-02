import torch
import config
from tqdm import tqdm
from cmath import inf
from utils import save_model


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
        train_loop.set_postfix(train_loss=loss.item())

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
                val_loop.set_postfix(val_loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    return avg_train_loss, avg_val_loss


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.least_loss = inf

    def __call__(self, model, validation_loss):
        if (validation_loss - self.least_loss) > self.min_delta:
            self.counter += 1
            print(
                f'Slowly losing patience ... ({self.counter}/{self.tolerance})\n')
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
            self.least_loss = validation_loss
            save_model(model.state_dict())
