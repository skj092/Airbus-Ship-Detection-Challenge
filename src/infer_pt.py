import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from tqdm import tqdm


valid_transforms = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class ShipSegmentData(Dataset):
    def __init__(self, image_path_list, img_dir, transforms=None):
        self.images = image_path_list
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, img_id)))

        # Apply transformations using Albumentations
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']

        return img

# function to convert mask to run-length encoding (RLE)


def mask_to_rle(mask):
    pixels = mask.flatten(order='F')  # Flatten the mask, column-wise
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get the change points
    runs[1::2] -= runs[::2]  # Calculate the lengths
    rle = ' '.join(str(x) for x in runs)
    return rle


def predict(img):
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img)
    out = model(img.unsqueeze(0).to(device))
    y = out.squeeze(0).squeeze(0).detach().cpu().numpy()
    return y


def prepare_rle_submission(mask, original_size=(768, 768)):
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Convert numpy array to PIL Image
    mask_img = Image.fromarray(mask * 255)  # PIL expects values in [0, 255]

    # Resize mask to original size
    resized_mask_img = mask_img.resize(original_size, resample=Image.NEAREST)

    # Convert back to numpy array and ensure binary
    resized_mask = np.array(resized_mask_img) > 127

    # Encode using RLE
    pixels = resized_mask.T.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    # Ensure the RLE starts with the first non-zero pixel
    if pixels[0] == 1:
        runs = np.concatenate([[0], runs])

    return ' '.join(str(x) for x in runs)


def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


ship_detection = pd.read_csv(
    '/kaggle/input/shipclassification/ship_detection.csv')
path = Path('/kaggle/input/airbus-ship-detection')
test_images = ship_detection.loc[ship_detection['p_ship'] > 0.5, [
    'id']]['id'].values.tolist()
test_names_nothing = ship_detection.loc[ship_detection['p_ship'] <= 0.5, [
    'id']]['id'].values.tolist()
print(len(test_images), len(test_names_nothing))

# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = Unet(encoder_name="resnet34", encoder_weights="imagenet",
             in_channels=3, classes=1).to(device)

model.load_state_dict(torch.load(
    "/kaggle/input/unet34-dice-0-87/best_model.pth", weights_only=False, map_location=device))
model.eval()

ship_list_dict = []
for name in test_names_nothing:
    ship_list_dict.append({'ImageId': name, 'EncodedPixels': np.nan})
df_t = pd.DataFrame(ship_list_dict)

rles = []

for i in tqdm(range(len(test_images))):
    orig_img = np.array(Image.open(
        os.path.join(path/'test_v2', test_images[i])))
    x = valid_transforms(image=orig_img)['image'].numpy().transpose(1, 2, 0)
    mask = predict(x)
    original_size = (768, 768)
    rle = prepare_rle_submission(mask, original_size)
    rles.append(rle)

ship_list_non_na_dict = []
for name, rle in zip(test_images, rles):
    ship_list_non_na_dict.append({'ImageId': name, 'EncodedPixels': rle})

df_v = pd.DataFrame(ship_list_non_na_dict)

submission = pd.concat([df_t, df_v])
submission.to_csv('submission.csv', index=False)
print(submission.head())
