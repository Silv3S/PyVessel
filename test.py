import uuid
import numpy as np
import torch
import config
from data_preparation import add_zero_padding
from dataset import get_dataloader
from metrics import SegmentationMetrics
from utils import list_directory, load_model
from visualize import save_graphical_confusion_matrix, plot_results_inline
from patchify import patchify, unpatchify


def evaluate_model(model, data_loader):
    segmentation_metrics = SegmentationMetrics()
    with torch.no_grad():
        for(_, (x, y)) in enumerate(data_loader):
            img = x.squeeze().numpy()
            img = np.moveaxis(img, 0, -1)
            mask = y.squeeze().numpy()
            image, padded_mask = add_zero_padding(
                x, y, format_NHWC=True)
            image_patches = patchify(
                image, (1, 3, config.PATCH_SIZE, config.PATCH_SIZE), step=config.PATCH_SIZE)
            # Patchify returns extra "1" dims, which can be squeezed
            image_patches = image_patches[0, 0, ...]
            preds_patches = np.zeros((
                image_patches.shape[0], image_patches.shape[1], config.PATCH_SIZE, config.PATCH_SIZE))

            for i in range(image_patches.shape[0]):
                for j in range(image_patches.shape[1]):
                    x = torch.from_numpy(image_patches[i, j, ...])
                    x = x.to(config.DEVICE)
                    pred = model(x)
                    pred = pred.squeeze().cpu().numpy()
                    preds_patches[i, j, ...] = (
                        (pred > 0.5) * 255).astype(np.uint8)
            reconstructed_image = unpatchify(
                preds_patches, np.squeeze(padded_mask).shape)
            prediction = reconstructed_image[0:mask.shape[0], 0:mask.shape[1]]

            plot_id = uuid.uuid4()
            plot_results_inline(img, mask, prediction, plot_id)
            save_graphical_confusion_matrix(mask, prediction, img, plot_id)
            segmentation_metrics.evaluate_pair(mask, prediction)
        segmentation_metrics.summary()


if __name__ == "__main__":
    model = config.MODEL_ARCHITECTURE.to(config.DEVICE)
    load_model(torch.load(config.BEST_MODEL_PATH), model)
    model.eval()

    test_image_paths, test_mask_paths = list_directory(
        config.TEST_DATASETS_PATH)
    test_loader = get_dataloader(test_image_paths, test_mask_paths, False, 1)

    evaluate_model(model, test_loader)
