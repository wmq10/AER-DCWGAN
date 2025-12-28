import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import DATASET_SPLITS_DIR, GENERATED_DATA_DIR
import os

COLOR_MAP = {

    'Bot': '#00202e',
    'Patator': '#003f5c',
    'DoS_Slowhttptest': '#2c4875',
    'Web_Attack': '#8a508f',
    'DoS_slowloris': '#bc5090',
    'DoS_GoldenEye': '#ff6361',
    'Infiltration': '#ff8531',
    'Heartbleed': '#ffa600'
}

plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def visualize_data_distribution():

    real_data_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
    if not os.path.exists(real_data_dir):

        return

    real_files = [f for f in os.listdir(real_data_dir) if f.endswith('.csv')]
    X_real = []
    y_real_type = []

    for file in real_files:
        attack_type = file.split('.')[0]
        real_path = os.path.join(real_data_dir, file)
        try:
            real_df = pd.read_csv(real_path)
        except Exception as e:

            continue
        if real_df.empty:

            continue
        if 'Label' not in real_df.columns:

            continue


        real_df['Label'] = real_df['Label'].astype(str).str.replace('_', ' ')

        X_real.append(real_df.drop('Label', axis=1).values)
        y_real_type.extend([attack_type] * len(real_df))

    if not X_real:

        return
    X_real = np.vstack(X_real)
    y_real_type = np.array(y_real_type)

    generated_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
    if not os.path.exists(generated_dir):

        return

    generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.csv')]
    X_generated = []
    y_gen_type = []

    for file in generated_files:
        attack_type = file.split('.')[0]
        generated_path = os.path.join(generated_dir, file)
        try:
            generated_df = pd.read_csv(generated_path)
        except Exception as e:

            continue
        if generated_df.empty:

            continue
        if 'Label' not in generated_df.columns:

            continue


        generated_df['Label'] = generated_df['Label'].astype(str).str.replace('_', ' ')

        X_generated.append(generated_df.drop('Label', axis=1).values)
        y_gen_type.extend([attack_type] * len(generated_df))

    if not X_generated:

        X_embedded_generated = np.array([])
        y_gen_type = np.array([])
    else:
        X_generated = np.vstack(X_generated)
        y_gen_type = np.array(y_gen_type)


    X_combined = np.vstack((X_real, X_generated)) if X_generated.size else X_real
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    try:
        X_embedded = tsne.fit_transform(X_combined)
    except Exception as e:

        return


    n_real = len(X_real)
    X_embedded_real = X_embedded[:n_real] if n_real > 0 else np.array([])
    X_embedded_generated = X_embedded[n_real:] if X_generated.size else np.array([])


    output_dir = os.path.join(".", "generated", "plots")
    os.makedirs(output_dir, exist_ok=True)


    plot_combined(
        X_embedded_real, y_real_type,
        X_embedded_generated, y_gen_type,
        COLOR_MAP, output_dir
    )


def plot_combined(X_real, y_real, X_generated, y_generated, color_map, output_dir):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for attack_type in color_map.keys():
        mask = (y_real == attack_type) if y_real.size else False
        if not mask.any():
            continue
        ax1.scatter(
            X_real[mask, 0], X_real[mask, 1],
            c=color_map[attack_type], marker='o', label=attack_type, alpha=0.6, s=30
        )
    ax1.set_title('Raw data distribution', fontsize=14)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    for attack_type in color_map.keys():
        mask = (y_generated == attack_type) if y_generated.size else False
        if not mask.any():
            continue
        ax2.scatter(
            X_generated[mask, 0], X_generated[mask, 1],
            c=color_map[attack_type], marker='s', label=attack_type, alpha=0.6, s=30
        )
    ax2.set_title('Generate data distributions', fontsize=14)
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Comparison of t-SNE visualizations of raw data vs. generated data', fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "tsne_combined.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


if __name__ == "__main__":
    visualize_data_distribution()