from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import config
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import list_directory


class RetinalBloodVesselsDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        if(transforms is None):
            self.transforms = A.Compose(
                [
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ],
            )
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        mask = np.array(Image.open(
            self.mask_paths[index]).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        augmentations = self.transforms(image=image, mask=mask)
        image = augmentations["image"]
        mask = augmentations["mask"]
        return image, mask


def get_train_dataloaders(dataset_name=None):
    image_paths, mask_paths = list_directory(config.PATCHES_PATH)

    if(dataset_name is not None):
        image_paths = [imp for imp in image_paths if (dataset_name in imp)]
        mask_paths = [mp for mp in mask_paths if (dataset_name in mp)]

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
            A.OneOf(
                [
                    A.CropAndPad(
                        percent=-0.3, sample_independently=True, p=0.25),
                    A.CropAndPad(
                        percent=-0.4, sample_independently=True, p=0.25),
                    A.CropAndPad(
                        percent=-0.2, sample_independently=False, p=0.25),
                    A.CropAndPad(
                        percent=-0.1, sample_independently=False, p=0.25),
                ],
                p=0.35),
            ToTensorV2(),
        ],
    )

    train_loader = get_dataloader(
        X_train, y_train, True, transforms=train_transform)
    val_loader = get_dataloader(X_val, y_val, False)
    return (train_loader, val_loader)


def get_dataloader(image_paths, mask_paths, shuffle, batch_size=config.BATCH_SIZE, transforms=None):
    dataset = RetinalBloodVesselsDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        transforms=transforms,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=shuffle,
    )
