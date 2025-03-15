#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from tqdm import tqdm

##############################################################################
# 1) Dataset Definition (Same as Training, but no label required in test CSV)
##############################################################################

class InferenceDataset(Dataset):
    """
    A Dataset for inference that:
      - Reads image filenames (IDs) from a CSV
      - Optionally, you might have a 'corr' column if you still want to compare, 
        but you don't need it for forward pass. If not present, it won't be used.
    """
    def __init__(self, img_dir, filenames, transform=None):
        self.img_dir = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Each image path: "images/<id>.png"
        img_path = os.path.join(self.img_dir, self.filenames[idx]) + '.png'
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


##############################################################################
# 2) Main Inference Code
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Inference script for ViT correlation model")
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/checkpoint_epoch_8.pt',
                        help="Path to the trained .pt checkpoint file")
    parser.add_argument('--test_csv', type=str, default='../csvs/test.csv',
                        help="CSV file with 'id' column listing image IDs (no .png extension) for inference")
    parser.add_argument('--img_dir', type=str, default='../images',
                        help="Directory containing the images (default: 'images')")
    parser.add_argument('--output_csv', type=str, default='../csvs/predictions.csv',
                        help="Output CSV file to store predictions (default: 'predictions.csv')")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for inference (default: 32)")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to run inference on (default: 'cuda:0' if available)")

    args = parser.parse_args()

    # Decide on device
    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {DEVICE}")

    # Read test CSV
    df_test = pd.read_csv(args.test_csv)
    # We assume there's a column named 'id'. If it's different, please adjust.
    test_ids = df_test['id'].values
    
    # Define the same image transform as training
    IMG_SIZE = 224
    test_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and loader
    test_dataset = InferenceDataset(
        img_dir=args.img_dir,
        filenames=test_ids,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,       # No shuffle in inference
        num_workers=2,
        pin_memory=True
    )

    # Create the same model architecture used in training
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)

    # Load checkpoint
    # If you used DataParallel in training, you might have saved model.module.state_dict().
    # Typically, you load it into the same model architecture here:
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    # If your checkpoint was saved with DataParallel, keys might start with "module."
    # Often, PyTorch will handle it automatically, but if not, you can do:
    #    from collections import OrderedDict
    #    new_state_dict = OrderedDict()
    #    for k, v in state_dict.items():
    #        name = k[7:] if k.startswith("module.") else k
    #        new_state_dict[name] = v
    #    model.load_state_dict(new_state_dict)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    predictions = []

    ############################################################################
    # Inference Loop
    ############################################################################
    with torch.no_grad():
        for batch_images in tqdm(test_loader, desc="Inference"):
            batch_images = batch_images.to(DEVICE)
            batch_outputs = model(batch_images).squeeze()  # shape: [B]
            # Convert to CPU and numpy
            batch_preds = batch_outputs.cpu().numpy()
            predictions.extend(batch_preds)

    # Store predictions in a DataFrame
    # The length of `predictions` should match the length of test_ids
    df_out = pd.DataFrame({
        'id': test_ids,
        'pred_corr': predictions
    })

    # Save to CSV
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}.")

if __name__ == "__main__":
    main()

