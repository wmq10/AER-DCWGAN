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

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载所有攻击类型（排除可能的异常文件）
train_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
attack_names = [
    f.split('.')[0] for f in os.listdir(train_dir)
    if f.endswith('.csv') and f.split('.')[0] in {'normal', 'DoS', 'Probe', 'R2L', 'U2R'}  # 限定5大攻击类型
]
print(f"检测到攻击类型: {attack_names}")

# 定义搜索空间（与之前一致）
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
        # 加载真实数据（位于 processed 子目录）
        real_path = os.path.join(DATASET_SPLITS_DIR, "processed", f"{attack_name}.csv")
        if not os.path.exists(real_path):
            print(f"警告: 真实数据文件不存在 - {real_path}")
            all_scores.append(0.0)
            continue

        attack_data = pd.read_csv(real_path)
        X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
        labels = attack_data['Label'].values
        label_map = joblib.load(LABEL_MAP_PATH)  # 使用 config 中定义的正确路径
        labels = np.array([label_map[label] for label in labels])
        reshaped_X = X_attack.reshape(-1, 1, 4, 6)
        reshaped_tensor = torch.from_numpy(reshaped_X).to(device)

        # 模型训练和生成数据（保持不变）
        ae_wgan = AE_WGAN_AER(
            img_shape=(1, 4, 6),
            num_classes=5,  # 指定类别数为5
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

        # 保存生成数据到正确目录（修正引号错误）
        generated_attacks_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
        os.makedirs(generated_attacks_dir, exist_ok=True)
        generated_path = os.path.join(generated_attacks_dir, f"{attack_name}.csv")  # 移除多余的右引号
        generated_df = pd.DataFrame(generated_samples.reshape(generated_samples.shape[0], -1))
        generated_df['Label'] = attack_name
        generated_df.to_csv(generated_path, index=False)

        # 计算得分（关键修改：real_dir 指向 processed 子目录）
        gs = GenerativeScores(model_name='AE-WGAN-AER', attack_names=[attack_name])
        scores = gs.calculate_scores(
            generated_dir=generated_attacks_dir,
            real_dir=os.path.join(DATASET_SPLITS_DIR, "processed")  # 修正为 processed 目录
        )

        # 防御性检查
        if attack_name not in scores:
            print(f"警告: 得分结果中缺少 {attack_name}，使用默认值")
            score = 0.0
        else:
            score = scores[attack_name]['cosine_similarity_ratio']

        all_scores.append(score)

    combined_score = np.min(all_scores)
    print(f"攻击类型得分: {all_scores}, 综合得分: {combined_score:.4f}")
    return -combined_score


# 运行贝叶斯优化
result = gp_minimize(
    objective, space, n_calls=10, random_state=0,
    acq_func='gp_hedge',  # 使用hedge策略平衡探索与利用
    n_points=10  # 初始随机采样点数
)

# 解析最优参数（基于综合得分）
best_index = np.argmax([-score for score in result.func_vals])  # 找到综合得分最高的迭代
best_params = result.x_iters[best_index]
best_combined_score = -result.func_vals[best_index]

# 输出各攻击类型在最优参数下的具体得分
print("\n最优参数下各攻击类型得分:")
final_scores = []
for attack_name in attack_names:
    # 重新评估最优参数在每个类型上的表现
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
        num_classes=5,  # 指定类别数为5
        latent_dim=best_params[0],
        recon_weight=best_params[1],
        gp_weight=best_params[2],
        aer_weight=best_params[3]
    ).to(device)
    ae_wgan.train(
        attack_name=attack_name,
        reshaped_X_attack=reshaped_tensor,
        labels=labels,
        epochs=1,  # 快速评估，仅训练1轮
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

# 输出结果
print(
    f"\n最优综合得分参数: latent_dim={best_params[0]}, recon_weight={best_params[1]:.2f}, gp_weight={best_params[2]:.2f}, aer_weight={best_params[3]:.2f}")
print(f"综合得分（最小得分）: {best_combined_score:.4f}")
for name, score in zip(attack_names, final_scores):
    print(f"{name} 得分: {score:.4f}")