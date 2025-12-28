import numpy as np
import pandas as pd
import os
from config import  DATASET_SPLITS_DIR, GENERATED_DATA_DIR, OUTPUT_DIR, LABEL_MAP_PATH
import joblib

def merge_datasets():

    merged_data_dir = os.path.join(OUTPUT_DIR, "merged_data")
    os.makedirs(merged_data_dir, exist_ok=True)


    train_path = os.path.join(DATASET_SPLITS_DIR, "train_processed.csv")


    if not os.path.exists(train_path):
        print(f" {train_path}")

    else:

        train_df = pd.read_csv(train_path)

        all_generated_dfs = []


        generated_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
        generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.csv')]

        for file in generated_files:
            generated_path = os.path.join(generated_dir, file)

            generated_df = pd.read_csv(generated_path)
            all_generated_dfs.append(generated_df)
        total_generated_df = pd.concat(all_generated_dfs, axis=0, ignore_index=True)

        total_merged_df = pd.concat([train_df, total_generated_df], axis=0, ignore_index=True)

        total_merged_path = os.path.join(OUTPUT_DIR, "total_merged_data.csv")
        total_merged_df.to_csv(total_merged_path, index=False)

if __name__ == "__main__":
    merge_datasets()