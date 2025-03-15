#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():
    CSV_PATH  = "../csvs/responses.csv"      # Adjust to your actual CSV path
    TRAIN_OUT = "../csvs/train.csv"
    TEST_OUT  = "../csvs/test.csv"
    TEST_RATIO = 0.1                # 10% for test

    # 1) Read CSV
    df = pd.read_csv(CSV_PATH)

    # 2) Sort by 'corr'
    df = df.sort_values('corr').reset_index(drop=True)

    # 3) Uniformly sample 10% across the sorted rows
    n = len(df)
    test_size = int(n * TEST_RATIO)
    # Create evenly spaced indices across the entire sorted DataFrame
    test_indices = np.linspace(0, n - 1, test_size, dtype=int)

    # 4) Extract test subset and the rest as train
    test_df = df.iloc[test_indices]

    # Create a mask to mark the train rows
    mask = np.ones(n, dtype=bool)
    mask[test_indices] = False
    train_df = df[mask]

    # 5) Save splits
    test_df.to_csv(TEST_OUT, index=False)
    train_df.to_csv(TRAIN_OUT, index=False)

    print(f"Total rows: {n}")
    print(f"Test rows: {len(test_df)} => saved to {TEST_OUT}")
    print(f"Train rows: {len(train_df)} => saved to {TRAIN_OUT}")

if __name__ == "__main__":
    main()

