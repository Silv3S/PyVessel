from sklearn.model_selection import train_test_split
import config
from dataset import RetinalBloodVesselsDataset
from utils import create_images_list, extract_patches
from imutils import paths
from torchvision import transforms

if(config.PREPARE_DATA):
    create_images_list()
    extract_patches()

image_paths = sorted(list(paths.list_images(config.PATCHED_IMAGES_PATH)))
mask_paths = sorted(list(paths.list_images(config.PATCHED_MASKS_PATH)))

(X_train, X_test, y_train, y_test) = train_test_split(image_paths, mask_paths,
                                                      test_size=config.TEST_SPLIT, random_state=config.RANDOM_SEED)

transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip()])

train_dataset = RetinalBloodVesselsDataset(X_train, X_test, transforms)

print(f"Created training dataset with {train_dataset.__len__()} images!")
