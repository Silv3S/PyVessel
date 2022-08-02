from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import config
from imutils import paths
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RetinalBloodVesselsDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        mask = np.array(Image.open(
            self.mask_paths[index]).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transforms is not None:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


def get_train_dataloaders(limits=0):
    image_paths = sorted(list(paths.list_images(config.PATCHES_PATH + "src/")))
    mask_paths = sorted(list(paths.list_images(config.PATCHES_PATH + "mask/")))

    if(limits != 0):
        image_paths = image_paths[:limits]
        mask_paths = mask_paths[:limits]

    (X_train, X_val, y_train, y_val) = train_test_split(image_paths, mask_paths,
                                                        test_size=config.VAL_SET_RATIO, random_state=config.RANDOM_SEED)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
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
    return (train_loader, val_loader)


def get_dataloader(image_paths, mask_paths, transforms, shuffle):
    dataset = RetinalBloodVesselsDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        transforms=transforms,
    )

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
    )
