import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from PIL import Image
from tqdm import tqdm

###############################################################################
# 1) Configuration
###############################################################################
CSV_PATH = '../../csvs/all_data.csv'  
IMG_DIR = '../../images'              
CHECKPOINT_DIR = '../../checkpoints'
KFOLDS = 10
BATCH_SIZE = 256
IMG_SIZE = 224
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOSS_THRESHOLD = 0.1  # Threshold for flagging samples
FLAGGED_DATA_CSV = '../../outputs/flagged_samples.csv'

###############################################################################
# 2) Dataset Definition
###############################################################################
class CorrelationDataset(Dataset):
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
        return image, label, img_id  # Return img_id for tracking

###############################################################################
# 3) Small CNN Model (from scratch)
###############################################################################
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.embedding_fc = nn.Linear(64 * 28 * 28, 128)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.embedding_fc(x))
        return self.regressor(x).squeeze(1)

###############################################################################
# 4) Main Training Logic with K-Fold Cross Validation
###############################################################################
def main():
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = CorrelationDataset(CSV_PATH, IMG_DIR, transform=transform)
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    flagged_data = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Starting Fold {fold+1}/{KFOLDS}")
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)
        
        model = SmallCNN().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            
            for images, labels, _ in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            
            train_loss = running_loss / len(train_loader.sampler)
            
            # Validation
            model.eval()
            val_loss_sum = 0.0
            
            with torch.no_grad():
                for images, labels, img_ids in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{NUM_EPOCHS} [VAL]"):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    preds = model(images)
                    losses = (preds - labels).abs().cpu().numpy()
                    
                    for img_id, true_corr, pred_corr, loss in zip(img_ids, labels.cpu().numpy(), preds.cpu().numpy(), losses):
                        if loss > LOSS_THRESHOLD:
                            flagged_data.append([img_id, true_corr, pred_corr, loss])
                    
                    val_loss_sum += criterion(preds, labels).item() * images.size(0)
                
            val_loss = val_loss_sum / len(val_loader.sampler)
            print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"cnn_fold_{fold+1}_epoch_{epoch+1}.pt"))
    
    # Save flagged data
    flagged_df = pd.DataFrame(flagged_data, columns=['id', 'true_corr', 'pred_corr', 'loss'])
    flagged_df.to_csv(FLAGGED_DATA_CSV, index=False)
    print(f"Flagged samples saved to {FLAGGED_DATA_CSV}")

if __name__ == "__main__":
    main()

