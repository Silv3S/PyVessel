import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from dataset import RetinalBloodVesselsDataset
from utils import read_txt_as_list
from visualize import plot_results_inline
from torchvision import transforms


def predict(model, data_loader):

    with torch.no_grad():
        for (_, (x, y)) in enumerate(data_loader):
            pred = model(x.to(config.DEVICE)).squeeze()
            pred = torch.sigmoid(pred)

            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            pred = pred.cpu().numpy()

            x = np.moveaxis(x, 0, -1)
            pred = (pred * 255).astype(np.uint8)

            plot_results_inline(x, y, pred)


if __name__ == "__main__":
    model = torch.load(config.BEST_MODEL_PATH)
    model.eval()

    basic_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.ToTensor()])
    test_dataset = RetinalBloodVesselsDataset(
        read_txt_as_list(config.TEST_IMAGES_PATH), read_txt_as_list(config.TEST_MASKS_PATH), basic_transforms)
    test_loader = DataLoader(test_dataset, shuffle=False)

    predict(model, test_loader)
