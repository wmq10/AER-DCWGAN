import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import DATASET_SPLITS_DIR, GENERATED_DATA_DIR
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
FONT_PATH = r"C:\Windows\Fonts\Arial.ttf"  # 请根据实际路径调整
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [os.path.basename(FONT_PATH)]  # 设置字体名称
plt.rcParams["font.serif"] = [os.path.basename(FONT_PATH)]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
def visualize_data_distribution():
    # 定义颜色映射
    COLOR_MAP = {
        'normal': '#003f5c',  # 蓝色
        'DoS': '#58508d',  # 红色
        'Probe': '#bc5090',  # 绿色
        'R2L': '#ff6361',  # 紫色
        'U2R': '#ffa600'  # 橙色
    }

    # ====================== 加载原始数据 ======================
    real_data_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
    real_files = [f for f in os.listdir(real_data_dir) if f.endswith('.csv')]

    X_real = []
    y_real_type = []  # 攻击类型标签

    for file in real_files:
        attack_type = file.split('.')[0]
        real_path = os.path.join(real_data_dir, file)
        real_df = pd.read_csv(real_path)
        X_real.append(real_df.drop('Label', axis=1).values)
        y_real_type.extend([attack_type] * len(real_df))

    X_real = np.vstack(X_real)
    y_real_type = np.array(y_real_type)

    # ====================== 加载生成数据 ======================
    generated_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
    generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.csv')]

    X_generated = []
    y_gen_type = []  # 攻击类型标签

    for file in generated_files:
        attack_type = file.split('.')[0]
        generated_path = os.path.join(generated_dir, file)
        generated_df = pd.read_csv(generated_path)
        X_generated.append(generated_df.drop('Label', axis=1).values)
        y_gen_type.extend([attack_type] * len(generated_df))

    X_generated = np.vstack(X_generated)
    y_gen_type = np.array(y_gen_type)

    # ====================== 合并数据并执行t-SNE降维 ======================
    print("正在执行t-SNE降维，这可能需要几分钟...")
    X_combined = np.vstack((X_real, X_generated))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_combined)

    # 分割降维结果
    n_real = len(X_real)
    X_embedded_real = X_embedded[:n_real]
    X_embedded_generated = X_embedded[n_real:]

    # 创建输出目录
    output_dir = os.path.join(".", "generated", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # ====================== 绘制并排对比图 ======================
    plot_combined(X_embedded_real, y_real_type, X_embedded_generated, y_gen_type, COLOR_MAP, output_dir)


def plot_combined(X_real, y_real, X_generated, y_generated, color_map, output_dir):
    """绘制并排对比图：左侧原始数据，右侧生成数据"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左侧：原始数据
    for attack_type in color_map.keys():
        mask = (y_real == attack_type)
        if np.sum(mask) > 0:
            ax1.scatter(
                X_real[mask, 0], X_real[mask, 1],
                c=color_map[attack_type], marker='o', label=attack_type, alpha=0.6, s=30
            )

    ax1.set_title('Raw data distribution', fontsize=14)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 右侧：生成数据
    for attack_type in color_map.keys():
        mask = (y_generated == attack_type)
        if np.sum(mask) > 0:
            ax2.scatter(
                X_generated[mask, 0], X_generated[mask, 1],
                c=color_map[attack_type], marker='s', label=attack_type, alpha=0.6, s=30
            )

    ax2.set_title('Generate data distributions', fontsize=14)
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 整体标题
    plt.suptitle('原始数据 vs 生成数据的 t-SNE 可视化对比', fontsize=16, y=1.02)
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(output_dir, "tsne_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    visualize_data_distribution()