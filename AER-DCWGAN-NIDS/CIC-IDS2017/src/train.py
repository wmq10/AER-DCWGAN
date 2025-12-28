
import torch
import time
import numpy as np
import pandas as pd
import warnings
from preprocessing import preprocess_cicids2017, split_dataset
from models import AE_WGAN_AER
from generative_scores import GenerativeScores
import os
from config import BATCH_SIZE, EPOCHS
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, DATASET_SPLITS_DIR, GENERATED_DATA_DIR, SCALER_PATH, LABEL_MAP_PATH, OUTPUT_DIR
import joblib
from merge_data import merge_datasets

from imblearn.over_sampling import SMOTE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def main():

    start_time = time.time()


    if not os.path.exists(PREPROCESSED_DATA_DIR) or not os.path.exists(SCALER_PATH):

        preprocess_cicids2017(RAW_DATA_DIR, DATASET_SPLITS_DIR)
        split_dataset(DATASET_SPLITS_DIR)



    try:
        scaler = joblib.load(SCALER_PATH)
        label_map = joblib.load(LABEL_MAP_PATH)
        reverse_label_map = {v: k for k, v in label_map.items()}

    except Exception as e:

        preprocess_cicids2017(RAW_DATA_DIR, DATASET_SPLITS_DIR)
        split_dataset(DATASET_SPLITS_DIR)
        scaler = joblib.load(SCALER_PATH)
        label_map = joblib.load(LABEL_MAP_PATH)
        reverse_label_map = {v: k for k, v in label_map.items()}

    train_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
    all_attacks = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')]


    for attack_idx, attack_name in enumerate(all_attacks):
        epoch_start = time.time()

        attack_path = os.path.join(train_dir, f"{attack_name}.csv")
        attack_data = pd.read_csv(attack_path)

        X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
        y_attack = attack_data['Label'].values
        existing_samples = len(attack_data)


        if existing_samples < 200:
            if len(np.unique(y_attack)) == 1:

                num_copies = 200 // existing_samples + 1
                X_attack = np.tile(X_attack, (num_copies, 1))[:200]
                y_attack = np.tile(y_attack, num_copies)[:200]
                existing_samples = len(X_attack)

            else:
                print(f"{existing_samples} < 200")


                smote = SMOTE(random_state=42, k_neighbors=min(5, existing_samples - 1))
                X_attack, y_attack = smote.fit_resample(X_attack, y_attack)
                existing_samples = len(X_attack)


        if attack_data.shape[1] != 31:
            raise ValueError(f": {attack_name} {attack_data.shape[1]} ")
        X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
        X_scaled = X_attack


        try:
            reshaped_X = X_scaled.reshape(-1, 1, 5, 6)
        except ValueError as e:

            raise
        reshaped_tensor = torch.from_numpy(reshaped_X).to(device)


        num_to_generate = 0


        if existing_samples > 80000:
            print(f"")
        else:
            if existing_samples < 10000:
                num_to_generate = 80000
            elif 10000 <= existing_samples < 30000:
                num_to_generate = 50000
            elif 30000 <= existing_samples < 50000:
                num_to_generate = 40000
            elif 50000 <= existing_samples < 100000:
                num_to_generate = 30000


            labels = attack_data['Label'].values

            labels = np.array([label_map[label] for label in labels])


        if num_to_generate > 0:
            print(f"")


            ae_wgan = AE_WGAN_AER(
                img_shape=(1, 5, 6),
                num_classes=12,
                latent_dim=32
            ).to(device)


            training_log = ae_wgan.train(
                attack_name=attack_name,
                reshaped_X_attack=reshaped_tensor,
                labels=labels,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )


            generated_samples = ae_wgan.generate_data(num_samples=num_to_generate)
            generated_denorm = generated_samples.reshape(-1, 30)
        else:

            generated_denorm = pd.DataFrame()

        save_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{attack_name}.csv")

        if generated_denorm.size != 0:
            generated_df = pd.DataFrame(generated_denorm)
            generated_df['Label'] = attack_name
            generated_df.to_csv(save_path, index=False)
            del generated_df
        else:
            print(f" {attack_name}")



    gs = GenerativeScores(model_name='AE-WGAN-AER', attack_names=all_attacks)
    scores = gs.calculate_scores(
        generated_dir=os.path.join(GENERATED_DATA_DIR, "generated_attacks"),
        real_dir=os.path.join(DATASET_SPLITS_DIR, "processed")
    )
    gs.save_scores(scores, output_dir=os.path.join(OUTPUT_DIR, "baseline_scores"))


    merge_datasets()


    total_time = (time.time() - start_time) / 60

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()