import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from dataset import RetinalBloodVesselsDataset
from metrics import SegmentationMetrics
from utils import load_model, read_txt_as_list
from visualize import plot_results_inline
from albumentations.pytorch import ToTensorV2
import albumentations as A


def predict(model, data_loader):
    segmentation_metrics = SegmentationMetrics()
    with torch.no_grad():
        for (_, (x, y)) in enumerate(data_loader):
            x = x.to(config.DEVICE)
            y = y.float().unsqueeze(1).to(config.DEVICE)
            pred = torch.sigmoid(model(x))
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()
            # CHW --> HWC
            x = np.moveaxis(x, 0, -1)
            pred = ((pred > 0.5) * 255).astype(np.uint8)
            plot_results_inline(x, y, pred)
            segmentation_metrics.evaluate_pair(y, pred)
    segmentation_metrics.summary()


if __name__ == "__main__":
    model = config.MODEL_ARCHITECTURE.to(config.DEVICE)
    load_model(torch.load(config.BEST_MODEL_PATH), model)
    model.eval()

    test_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_dataset = RetinalBloodVesselsDataset(
        image_paths=read_txt_as_list(config.TEST_IMAGES_PATH),
        mask_paths=read_txt_as_list(config.TEST_MASKS_PATH),
        transforms=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    predict(model, test_loader)
