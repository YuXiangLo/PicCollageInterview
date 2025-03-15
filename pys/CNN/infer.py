#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

###############################################################################
# 1) Model Definition (same as in train.py)
###############################################################################
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 poolings on a 224x224 image, we get 28x28 with 64 channels:
        # => 64*28*28 = 50176 => map to 128
        self.embedding_fc = nn.Linear(64 * 28 * 28, 128)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)           # Flatten
        x = F.relu(self.embedding_fc(x))    # 128-dim
        out = self.regressor(x).squeeze(1)  # scalar output
        return out


###############################################################################
# 2) Dataset for Inference
###############################################################################
class InferenceDataset(Dataset):
    """
    Loads images from a CSV that has at least 'id' (and optionally 'corr').
    We'll ignore 'corr' for forward pass, but you can store it if you want to
    compare predictions with ground truth.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Check if ground-truth correlation exists
        self.has_corr = 'corr' in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # If you want to keep the ground truth in the item (for later comparison):
        if self.has_corr:
            corr_value = row['corr']
            return image, img_id, corr_value
        else:
            return image, img_id


###############################################################################
# 3) Main Inference Logic
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Inference script for CNN from scratch")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained CNN checkpoint (e.g., '../../checkpoints/cnn_checkpoint_epoch_10.pt')")
    parser.add_argument('--test_csv', type=str, required=True,
                        help="CSV file with 'id' (and optionally 'corr') for test/inference")
    parser.add_argument('--img_dir', type=str, default='../../images',
                        help="Directory with .png images (default ../../images)")
    parser.add_argument('--output_csv', type=str, default='../../outputs/cnn_predictions.csv',
                        help="Where to save predictions (default ../../outputs/cnn_predictions.csv)")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for inference (default=64)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device (cuda or cpu). Default uses cuda if available.")
    args = parser.parse_args()

    # Decide on device
    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Inference on device: {DEVICE}")

    # 3A) Build the same CNN model
    model = SmallCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 3B) Create Dataset / Dataloader with same transforms used in training
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    infer_dataset = InferenceDataset(args.test_csv, args.img_dir, transform=transform)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

    predictions = []
    ids = []
    has_corr = infer_dataset.has_corr
    gt_corrs = []  # if ground truth is available

    # 3C) Forward pass, collecting predictions
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Inference"):
            if has_corr:
                images, batch_ids, batch_corr = batch
            else:
                images, batch_ids = batch

            images = images.to(DEVICE)
            batch_preds = model(images)  # shape [B]
            batch_preds = batch_preds.cpu().numpy()  # move to CPU

            predictions.extend(batch_preds)
            ids.extend(batch_ids)

            if has_corr:
                gt_corrs.extend(batch_corr.numpy())

    # 3D) Build output DataFrame
    if has_corr:
        df_out = pd.DataFrame({
            'id': ids,
            'true_corr': gt_corrs,
            'pred_corr': predictions
        })
    else:
        df_out = pd.DataFrame({
            'id': ids,
            'pred_corr': predictions
        })

    # 3E) Save results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()

