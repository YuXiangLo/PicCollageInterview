#!/usr/bin/env python3
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

###############################################################################
# 1) Configuration
###############################################################################
CSV_PATH = '../../csvs/train.csv'   # CSV with ['id', 'corr']
IMG_DIR = '../../images'            # Directory with images: id + '.png'
CHECKPOINT_DIR = '../../checkpoints'
VAL_SPLIT = 0.1
BATCH_SIZE = 256
IMG_SIZE = 224
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
# 2) Dataset Definition
###############################################################################
class CorrelationDataset(Dataset):
    """Loads (image, correlation) from CSV and directory."""
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        corr_value = row['corr']

        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(corr_value, dtype=torch.float)
        return image, label


###############################################################################
# 3) Small CNN Model (from scratch)
###############################################################################
class SmallCNN(nn.Module):
    """
    A simple CNN from scratch. We'll have:
    - 3 conv layers
    - Flatten
    - A fully connected 'embedding' layer
    - A final linear layer to output correlation (1D)
    """
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # We'll produce an embedding of size 128 from the final feature map
        # after the 3 conv layers + pooling. The input image is 224x224,
        # so after 3 poolings, it becomes 224->112->56->28 => 28x28.
        # Channels: 64, so feature map is [64, 28, 28] => 64*28*28 = 50176.
        # We'll map that to 128-dim.
        self.embedding_fc = nn.Linear(64 * 28 * 28, 128)

        # Final linear layer to get 1 output (correlation)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        # Conv + ReLU + Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)  # shape [B, 64*28*28]
        
        # Map to 128-dim embedding
        x = F.relu(self.embedding_fc(x))  # shape [B, 128]
        
        # Finally, get a single correlation output
        out = self.regressor(x)  # shape [B, 1]
        return out.squeeze(1)    # shape [B]


###############################################################################
# 4) Main Training Logic
###############################################################################
def main():
    # --- A) Prepare Dataset + Data Loaders ---
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        # If you like, you can still do ImageNet normalization or skip it:
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = CorrelationDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataset_size = len(full_dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Total: {dataset_size} images. Train: {train_size}, Val: {val_size}")

    # --- B) Build Model + Optimizer ---
    model = SmallCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- C) Train ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)  # shape [B]
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [VAL]"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                preds = model(images)
                val_loss_sum += criterion(preds, labels).item() * images.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} => "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint each epoch
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"cnn_checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

