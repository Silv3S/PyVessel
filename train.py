import torch
import config
from tqdm import tqdm


def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler):
    train_loss = 0
    val_loss = 0
    train_loop = tqdm(train_loader)
    val_loop = tqdm(val_loader)

    model.train()
    for batch_idx, (x, y) in enumerate(train_loop):
        x = x.to(device=config.DEVICE)
        y = y.float().unsqueeze(1).to(device=config.DEVICE)
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss
        train_loop.set_postfix(loss=loss.item())

    # Validation, turn off for faster training
    if config.USE_VALIDATION_SET:
        with torch.no_grad():
            model.eval()

            for batch_idx, (x, y) in enumerate(val_loop):
                x = x.to(device=config.DEVICE)
                y = y.float().unsqueeze(1).to(device=config.DEVICE)
                predictions = model(x)
                loss += loss_fn(predictions, y)
                val_loss += loss
                val_loop.set_postfix(loss=loss.item())

    return train_loss.cpu().detach().numpy(), val_loss.cpu().detach().numpy()
