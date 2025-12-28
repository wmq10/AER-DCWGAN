import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer
from models import AE_WGAN_AER
from generative_scores import GenerativeScores
import numpy as np
import pandas as pd
import os
from config import DATASET_SPLITS_DIR, GENERATED_DATA_DIR, OUTPUT_DIR, SCALER_PATH,LABEL_MAP_PATH
import joblib
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
attack_names = [
    f.split('.')[0] for f in os.listdir(train_dir)
    if f.endswith('.csv') and f.split('.')[0] in {'normal', 'DoS', 'Probe', 'R2L', 'U2R'}
]
print(f"{attack_names}")


space = [
    Integer(20, 80, name='latent_dim'),
    Real(2, 10, name='recon_weight'),
    Real(5, 10, name='gp_weight'),
    Real(0.6, 3, name='aer_weight')
]


def objective(params):
    latent_dim, recon_weight, gp_weight, aer_weight = params
    all_scores = []

    for attack_name in attack_names:

        real_path = os.path.join(DATASET_SPLITS_DIR, "processed", f"{attack_name}.csv")
        if not os.path.exists(real_path):

            all_scores.append(0.0)
            continue

        attack_data = pd.read_csv(real_path)
        X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
        labels = attack_data['Label'].values
        label_map = joblib.load(LABEL_MAP_PATH)
        labels = np.array([label_map[label] for label in labels])
        reshaped_X = X_attack.reshape(-1, 1, 4, 6)
        reshaped_tensor = torch.from_numpy(reshaped_X).to(device)


        ae_wgan = AE_WGAN_AER(
            img_shape=(1, 4, 6),
            num_classes=5,
            latent_dim=latent_dim,
            recon_weight=recon_weight,
            gp_weight=gp_weight,
            aer_weight=aer_weight
        ).to(device)

        ae_wgan.train(
            attack_name=attack_name,
            reshaped_X_attack=reshaped_tensor,
            labels=labels,
            epochs=10,
            batch_size=64
        )

        generated_samples = ae_wgan.generate_data(num_samples=1000)
        if isinstance(generated_samples, torch.Tensor):
            generated_samples = generated_samples.cpu().numpy()


        generated_attacks_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
        os.makedirs(generated_attacks_dir, exist_ok=True)
        generated_path = os.path.join(generated_attacks_dir, f"{attack_name}.csv")
        generated_df = pd.DataFrame(generated_samples.reshape(generated_samples.shape[0], -1))
        generated_df['Label'] = attack_name
        generated_df.to_csv(generated_path, index=False)


        gs = GenerativeScores(model_name='AE-WGAN-AER', attack_names=[attack_name])
        scores = gs.calculate_scores(
            generated_dir=generated_attacks_dir,
            real_dir=os.path.join(DATASET_SPLITS_DIR, "processed")
        )


        if attack_name not in scores:

            score = 0.0
        else:
            score = scores[attack_name]['cosine_similarity_ratio']

        all_scores.append(score)

    combined_score = np.min(all_scores)

    return -combined_score



result = gp_minimize(
    objective, space, n_calls=10, random_state=0,
    acq_func='gp_hedge',
    n_points=10
)


best_index = np.argmax([-score for score in result.func_vals])
best_params = result.x_iters[best_index]
best_combined_score = -result.func_vals[best_index]


final_scores = []
for attack_name in attack_names:

    attack_path = os.path.join(train_dir, f"{attack_name}.csv")
    attack_data = pd.read_csv(attack_path)
    X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
    labels = attack_data['Label'].values
    label_map = joblib.load(LABEL_MAP_PATH)
    labels = np.array([label_map[label] for label in labels])
    reshaped_X = X_attack.reshape(-1, 1, 4, 6)
    reshaped_tensor = torch.from_numpy(reshaped_X).to(device)

    ae_wgan = AE_WGAN_AER(
        img_shape=(1, 4, 6),
        num_classes=5,
        latent_dim=best_params[0],
        recon_weight=best_params[1],
        gp_weight=best_params[2],
        aer_weight=best_params[3]
    ).to(device)
    ae_wgan.train(
        attack_name=attack_name,
        reshaped_X_attack=reshaped_tensor,
        labels=labels,
        epochs=1,
        batch_size=64
    )
    generated_samples = ae_wgan.generate_data(num_samples=1000)
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()

    gs = GenerativeScores(model_name='AE-WGAN-AER', attack_names=[attack_name])
    scores = gs.calculate_scores(
        generated_dir=GENERATED_DATA_DIR,
        real_dir=DATASET_SPLITS_DIR
    )
    final_scores.append(scores[attack_name]['cosine_similarity_ratio'])


for name, score in zip(attack_names, final_scores):
