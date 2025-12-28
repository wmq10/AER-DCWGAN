import torch
import numpy as np
import pandas as pd
from models import AE_WGAN_AER
from generative_scores import GenerativeScores
from skopt import gp_minimize
from skopt.space import Real, Integer
from config import DATASET_SPLITS_DIR, GENERATED_DATA_DIR, OUTPUT_DIR
from config import LATENT_DIM_MIN, LATENT_DIM_MAX, RECON_WEIGHT_MIN, RECON_WEIGHT_MAX, GP_WEIGHT_MIN, GP_WEIGHT_MAX, AER_WEIGHT_MIN, AER_WEIGHT_MAX
import os


train_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
attack_name = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')][0]
attack_path = os.path.join(train_dir, f"{attack_name}.csv")
attack_data = pd.read_csv(attack_path)
X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
X_scaled = X_attack
reshaped_X = X_scaled.reshape(-1, 3, 6, 1)
reshaped_tensor = torch.from_numpy(reshaped_X)
labels = attack_data['Label'].values

# 定义评估函数
def evaluate_ae_wgan(params):
    latent_dim, recon_weight, gp_weight, aer_weight = params
    ae_wgan = AE_WGAN_AER(
        img_shape=(4, 9, 1),
        latent_dim=int(latent_dim),
        recon_weight=recon_weight,
        gp_weight=gp_weight,
        aer_weight=aer_weight
    )

    ae_wgan.train(
        attack_name=attack_name,
        reshaped_X_attack=reshaped_tensor,
        epochs=10,
        batch_size=64
    )
    gs = GenerativeScores(model_name='AE-WGAN-AER', attack_names=[attack_name])
    scores = gs.calculate_scores(
        generated_dir=os.path.join(GENERATED_DATA_DIR, "generated_attacks"),
        real_dir=os.path.join(DATASET_SPLITS_DIR, "processed")
    )
    print("scores:", scores)
    try:
        return -scores[attack_name]['diversity']
    except KeyError:
        print(f"Key 'diversity' not found in scores for attack {attack_name}.")
        return float('inf')


search_space = [
    Integer(LATENT_DIM_MIN, LATENT_DIM_MAX, name='latent_dim'),
    Real(RECON_WEIGHT_MIN, RECON_WEIGHT_MAX, name='recon_weight'),
    Real(GP_WEIGHT_MIN, GP_WEIGHT_MAX, name='gp_weight'),
    Real(AER_WEIGHT_MIN, AER_WEIGHT_MAX, name='aer_weight')
]


result = gp_minimize(evaluate_ae_wgan, search_space, n_calls=10)

