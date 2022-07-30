from cmath import inf
import time
from sklearn.model_selection import train_test_split
import torch
import config
from data_preparation import extract_train_patches, split_train_test_images
from dataset import get_dataloader
from train import train_fn
from utils import clear_image_directories, load_model, save_model
from imutils import paths
from torch.optim import Adam
from visualize import plot_loss_history
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam


if __name__ == '__main__':
    if(config.PREPARE_DATASETS):
        clear_image_directories()
        split_train_test_images()
        extract_train_patches()

    image_paths = sorted(list(paths.list_images(config.PATCHES_PATH + "src/")))
    mask_paths = sorted(list(paths.list_images(config.PATCHES_PATH + "mask/")))

    # Limit training until application is working correctly
    image_paths = image_paths[:200]
    mask_paths = mask_paths[:200]

    (X_train, X_val, y_train, y_val) = train_test_split(image_paths, mask_paths,
                                                        test_size=config.VAL_SET_RATIO, random_state=config.RANDOM_SEED)
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader = get_dataloader(X_train, y_train, train_transform, True)
    val_loader = get_dataloader(X_val, y_val, val_transforms, False)

    device = config.DEVICE
    model = config.MODEL_ARCHITECTURE.to(device)
    loss_func = config.LOSS_FUNC
    opt = Adam(model.parameters(), config.LR)

    if config.LOAD_PRETRAINED_MODEL:
        load_model(torch.load(config.BEST_MODEL_PATH), model)

    scaler = torch.cuda.amp.GradScaler()
    least_loss = inf
    train_stats = {'train_loss': [], 'val_loss': []}

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    start_time = time.time()
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch: {epoch + 1} of {config.NUM_EPOCHS}')
        train_loss, val_loss = train_fn(
            train_loader, val_loader, model, opt, loss_func, scaler)

        train_stats['train_loss'].append(train_loss)
        train_stats['val_loss'].append(val_loss)

        if val_loss < least_loss:
            least_loss = val_loss
            save_model(model.state_dict())

    print('That\'s all Folks!')
    print(f'Total training time: {(time.time() - start_time):.2f}s')
    plot_loss_history(train_stats)
