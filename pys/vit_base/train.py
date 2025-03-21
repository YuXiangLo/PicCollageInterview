import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import timm

# For progress bar
from tqdm import tqdm

# 1) Configuration
CSV_PATH = '../../csvs/train.csv'   
IMG_DIR = '../../images'            
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.1  
IMG_SIZE = 224   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2) Read CSV and prepare list of (path, correlation)
df = pd.read_csv(CSV_PATH)
all_filenames = df['id'].values
all_labels = df['corr'].values.astype(float)

# 3) Define a Dataset that loads images and their correlation
class CorrelationDataset(Dataset):
    def __init__(self, img_dir, filenames, labels, transform=None):
        self.img_dir = img_dir
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx]) + '.png'
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)

# 4) Transforms (Resize, ToTensor, Normalize).
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# 5) Create the full dataset, then split train/val
full_dataset = CorrelationDataset(IMG_DIR, all_filenames, all_labels, transform=train_transform)
dataset_size = len(full_dataset)
val_size = int(VAL_SPLIT * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 6) Create a ViT-Base model from timm, set num_classes=1 for regression
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
model = model.to(DEVICE)

# 7) Define Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 8) Training Loop with tqdm
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    # --- TRAINING PASS ---
    train_batches = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze()  # shape [batch_size]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_batches += 1
    
    epoch_loss = running_loss / len(train_loader.dataset)

    # --- VALIDATION PASS ---
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [VAL]"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_batches += 1

    val_loss = val_loss_sum / len(val_loader.dataset)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} => "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}")

    # torch.save(model.state_dict(), f"../../checkpoints/vit_checkpoint_epoch_{epoch + 1}.pt")


