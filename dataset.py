from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import config


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
