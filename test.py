import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from data_preparation import add_zero_padding
from dataset import RetinalBloodVesselsDataset
from metrics import SegmentationMetrics
from utils import load_model
from visualize import plot_results_inline
from albumentations.pytorch import ToTensorV2
import albumentations as A
from imutils import paths
from patchify import patchify, unpatchify


def predict(model, data_loader):
    segmentation_metrics = SegmentationMetrics()
    with torch.no_grad():
        for(_, (x, y)) in enumerate(data_loader):
            img = x.squeeze().numpy()
            img = np.moveaxis(img, 0, -1)
            mask = y.squeeze().numpy()
            image, padded_mask = add_zero_padding(
                x, y, format_NHWC=True)
            image_patches = patchify(
                image, config.PATCH_SHAPE_IMG_NHWC, step=config.PATCH_STEP)
            # Patchify returns extra "1" dims, which can be squeezed
            image_patches = image_patches[0, 0, ...]
            preds_patches = np.zeros((
                image_patches.shape[0], image_patches.shape[1], config.PATCH_SHAPE_MASK[0], config.PATCH_SHAPE_MASK[1]))

            for i in range(image_patches.shape[0]):
                for j in range(image_patches.shape[1]):
                    x = torch.from_numpy(image_patches[i, j, ...])
                    x = x.to(config.DEVICE)
                    pred = torch.sigmoid(model(x))
                    pred = pred.squeeze().cpu().numpy()
                    preds_patches[i, j, ...] = (
                        (pred > 0.5) * 255).astype(np.uint8)
            reconstructed_image = unpatchify(
                preds_patches, np.squeeze(padded_mask).shape)
            prediction = reconstructed_image[0:mask.shape[0], 0:mask.shape[1]]

            plot_results_inline(img, mask, prediction)
            segmentation_metrics.evaluate_pair(mask, prediction)
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
        image_paths=sorted(
            list(paths.list_images(config.TEST_DATASETS_PATH + "src/"))),
        mask_paths=sorted(
            list(paths.list_images(config.TEST_DATASETS_PATH + "mask/"))),
        transforms=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    predict(model, test_loader)
