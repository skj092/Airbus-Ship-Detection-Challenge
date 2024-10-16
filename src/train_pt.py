import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt


# Custom function to convert run-length encoding to mask
def get_mask(image_id, df):
    rows = df.loc[df['ImageId'] == image_id]
    shape = (768, 768)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    masks = rows['EncodedPixels'].tolist()
    if not masks:
        return img.reshape(shape)
    for mask in masks:
        s = mask.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1
    return img.reshape(shape).T


# Function to convert mask to run-length encoding (RLE)
def mask_to_rle(mask):
    """
    Convert a mask (2D array) to run-length encoding (RLE) as required for submission.

    Args:
        mask (np.array): Mask with shape (H, W) where 1 = ship, 0 = background.

    Returns:
        str: RLE encoded mask.
    """
    pixels = mask.flatten(order='F')  # Flatten the mask, column-wise
    # Add leading and trailing zeros
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get the change points
    runs[1::2] -= runs[::2]  # Calculate the lengths
    rle = ' '.join(str(x) for x in runs)
    return rle


def generate_submission(df, image_ids, masks):
    """
    Generate a submission DataFrame from predicted masks and corresponding image IDs.
    Args:
        df (pd.DataFrame): Original DataFrame with 'ImageId' column.
        image_ids (list): List of image IDs for which masks were predicted.
        masks (list): List of predicted masks corresponding to the image IDs.

    Returns:
        pd.DataFrame: DataFrame with 'ImageId' and 'EncodedPixels' for submission.
    """
    submissions = []
    for image_id, mask in zip(image_ids, masks):
        rle = mask_to_rle(mask)
        submissions.append({'ImageId': image_id, 'EncodedPixels': rle})

    submission_df = pd.DataFrame(submissions, columns=['ImageId', 'EncodedPixels'])
    return submission_df


def get_preprocessing_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def predict(img):
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img)
    out = model(img.unsqueeze(0).to('cuda'))
    y = out.squeeze(0).squeeze(0).detach().cpu().numpy()
    return y


def plot_img_mask(img_path):

    f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True, figsize=(30, 10))

    # Original Imaeg without augmentation
    orig_img = np.array(Image.open(os.path.join(path/'train_v2', img_path)))
    ax0.imshow(orig_img)  # Convert BGR to RGB for correct color display
    ax0.set_title('Original Image')
    ax0.set_axis_off()

    # Preprocessing
    x = valid_transforms(image=orig_img)['image'].numpy().transpose(1, 2, 0)
    ax1.imshow(x)  # Convert BGR to RGB for correct color display
    ax1.set_title('Augmented Image')
    ax1.set_axis_off()

    # predict the mask
    mask = predict(x)

    # Display the mask in ax2
    ax2.imshow(mask, cmap='gray')  # Assuming mask is a grayscale image
    ax2.set_title('Mask')
    ax2.set_axis_off()

    # Overlay mask on the image in ax3
    ax3.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))  # Display original image
    ax3.imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask with transparency
    ax3.set_title('Image with Mask')
    ax3.set_axis_off()

    plt.show()


# Custom Dataset class
class ShipSegmentData(Dataset):
    def __init__(self, image_path_list, img_dir, df, transforms=None):
        self.images = image_path_list
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = df

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, img_id)))

        # Get mask
        mask = get_mask(img_id, self.df)

        # Apply transformations using Albumentations
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Ensure the mask has a channel dimension (1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(
            0)  # Add channel dimension for mask

        return img, mask


# Define the training function
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, masks in tqdm(dataloader):
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


# Define the validation function
def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, masks in tqdm(dataloader):
            inputs, masks = inputs.to(device), masks.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(dataloader)


def cut_empty(names):
    return [name for name in names if (type(df.loc[name]['EncodedPixels']) != float)]


if __name__ == "__main__":
    # Data preparation
    path = Path('/home/sonujha/rnd/Airbus-Ship-Detection-Challenge/data/')
    df = pd.read_csv(path / 'train_ship_segmentations_v2.csv')
    df.set_index('ImageId', inplace=True)

    all_train_images = [f for f in os.listdir(path / 'train_v2')][:500]

    all_train_images = cut_empty(all_train_images)
    train_images, valid_images = train_test_split(
        all_train_images, test_size=0.2, random_state=42)
    df.reset_index(inplace=True)

    print(f"Total train images: {len(train_images)}")
    print(f"Total valid images: {len(valid_images)}")

    # Define Albumentations transforms
    train_transforms = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Datasets and DataLoaders
    train_ds = ShipSegmentData(
        train_images, path / 'train_v2', df, transforms=train_transforms)
    valid_ds = ShipSegmentData(
        valid_images, path / 'train_v2', df, transforms=valid_transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False)

    # Define device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = Unet(encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=1).to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss with logits
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and Validation Loop
    num_epochs = 10
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train the model
        train_loss = train_model(
            model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate the model
        val_loss = validate_model(model, valid_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

    print("Training completed!")
