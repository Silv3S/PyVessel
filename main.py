import time
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import config
from dataset import RetinalBloodVesselsDataset
from utils import create_images_list, extract_patches, save_model, save_to_file
from imutils import paths
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from visualize import plot_loss_history

if(config.PREPARE_DATA):
    create_images_list()
    extract_patches()

image_paths = sorted(list(paths.list_images(config.PATCHED_IMAGES_PATH)))
mask_paths = sorted(list(paths.list_images(config.PATCHED_MASKS_PATH)))

(X_train_, X_test, y_train_, y_test) = train_test_split(image_paths, mask_paths,
                                                        test_size=config.TEST_SPLIT, random_state=config.RANDOM_SEED)
(X_train, X_val, y_train, y_val) = train_test_split(X_train_, y_train_,
                                                    test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED)
save_to_file(config.TEST_IMAGES_PATH, X_test)
save_to_file(config.TEST_MASKS_PATH, y_test)

basic_transforms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.ToTensor()])

augmentations = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()])

train_dataset = RetinalBloodVesselsDataset(X_train, y_train, augmentations)
val_dataset = RetinalBloodVesselsDataset(X_val, y_val, basic_transforms)
test_dataset = RetinalBloodVesselsDataset(X_test, y_test, basic_transforms)

print(f'Dataset division:\n- Training set: {train_dataset.__len__()} patches \n- Validation set: '
      f'{val_dataset.__len__()} patches\n- Test set: {test_dataset.__len__()} patches')

train_loader = DataLoader(train_dataset, shuffle=True,
                          batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
val_loader = DataLoader(val_dataset, shuffle=False,
                        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

device = config.DEVICE
model = config.MODEL_ARCHITECTURE.to(device)
loss_func = config.LOSS_FUNC
opt = Adam(model.parameters(), config.LR)

train_steps = train_dataset.__len__() // config.BATCH_SIZE
val_steps = val_dataset.__len__() // config.BATCH_SIZE
train_stats = {'train_loss': [], 'validation_loss': []}

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
start_time = time.time()

for epoch in tqdm(range(config.NUM_EPOCHS)):
    print(f'Epoch: {epoch + 1} of {config.NUM_EPOCHS}')
    model.train()
    train_loss = 0

    for (i, (x, y)) in enumerate(train_loader):
        (x, y) = (x.to(device), y.to(device))
        pred = model(x)
        loss = loss_func(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss

    train_loss = train_loss / train_steps
    train_stats['train_loss'].append(train_loss.cpu().detach().numpy())
    print(f'Training loss: {train_loss:.4f}')

    # Validation, turn off for faster training
    if config.USE_VALIDATION_SET:
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for (x, y) in val_loader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                val_loss += loss_func(pred, y)
        val_loss = val_loss / val_steps
        train_stats['validation_loss'].append(val_loss.cpu().detach().numpy())
        print(f'Validation loss: {val_loss:.4f}')

print('That\'s all Folks!')
print(f'Total training time: {(time.time() - start_time):.2f}s')

save_model(model)
plot_loss_history(train_stats)
