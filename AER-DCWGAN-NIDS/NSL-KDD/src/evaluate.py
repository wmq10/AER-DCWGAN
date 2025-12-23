import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import joblib
from config import SCALER_PATH
from sklearn.preprocessing import MinMaxScaler
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, DATASET_SPLITS_DIR, GENERATED_DATA_DIR, OUTPUT_DIR  # 导入配置文件

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenerativeScores:
    def __init__(self, model_name, attack_names):
        self.model_name = model_name
        self.attack_names = attack_names

    def cosine_similarity(self, generated_data, real_data, batch_size=2048):
        """计算生成数据与真实数据的余弦相似度"""
        min_length = min(generated_data.shape[0], real_data.shape[0])
        generated_data = generated_data[:min_length]
        real_data = real_data[:min_length]

        similarities = []
        for i in range(0, min_length, batch_size):
            batch_gen = torch.tensor(generated_data[i:i + batch_size], device=device)
            batch_real = torch.tensor(real_data[i:i + batch_size], device=device)
            similarities.append(nn.CosineSimilarity(dim=1)(batch_gen, batch_real).mean().item())
        return np.mean(similarities)

    def real_self_cosine_similarity(self, real_data, sample_pairs=1000, batch_size=512):
        """计算真实数据内部样本对的余弦相似度（自相似度基线）"""
        num_samples = real_data.shape[0]
        if num_samples < 2:
            raise ValueError("真实数据样本数不足，无法计算自相似度")

        # 随机选择样本对
        idx1 = np.random.randint(0, num_samples, size=sample_pairs)
        idx2 = np.random.randint(0, num_samples, size=sample_pairs)

        similarities = []
        for i in range(0, sample_pairs, batch_size):
            batch_idx1 = idx1[i:i + batch_size]
            batch_idx2 = idx2[i:i + batch_size]

            batch1 = torch.tensor(real_data[batch_idx1], device=device)
            batch2 = torch.tensor(real_data[batch_idx2], device=device)

            sim = nn.CosineSimilarity(dim=1)(batch1, batch2)
            similarities.append(sim.cpu().numpy())

        return np.mean(np.concatenate(similarities))

    def calculate_scores(self, generated_dir, real_dir):
        """计算并比较生成数据与真实数据的相似度和MMD"""
        scaler = joblib.load(SCALER_PATH)
        scores = {}

        # 动态获取真实攻击名称
        real_attack_names = [
            f.split('.')[0] for f in os.listdir(real_dir)
            if f.endswith('.csv') and 'label' not in f.lower()
        ]

        for attack_name in real_attack_names:
            generated_path = os.path.join(generated_dir, f"{attack_name}.csv")
            real_path = os.path.join(real_dir, f"{attack_name}.csv")

            if not os.path.exists(generated_path) or not os.path.exists(real_path):
                print(f"文件不存在: {generated_path} 或 {real_path}")
                continue

            # 加载数据并统一删除标签列
            generated_data = pd.read_csv(generated_path)
            real_data = pd.read_csv(real_path)
            label_cols = [col for col in generated_data.columns if col.lower() == 'label']
            generated_data = generated_data.drop(label_cols, axis=1).astype(np.float32).values
            real_data = real_data.drop(label_cols, axis=1).astype(np.float32).values

            # 计算自相似度基线
            real_self_sim = self.real_self_cosine_similarity(real_data)


            # 计算生成数据与真实数据的相似度
            gen_real_sim = self.cosine_similarity(generated_data, real_data)


            # 计算相对得分（生成数据与自相似度基线的比例）
            sim_ratio = gen_real_sim / real_self_sim if real_self_sim > 0 else 0

            scores[attack_name] = {
                "real_self_cosine_sim": real_self_sim,
                "gen_real_cosine_sim": gen_real_sim,
                "cosine_similarity_ratio": sim_ratio,
                       }

            # 打印详细结果
            print(f"\n[{attack_name}] 评估结果:")
            print(f"  真实数据自余弦相似度: {real_self_sim:.4f}")
            print(f"  生成数据与真实数据余弦相似度: {gen_real_sim:.4f}")
            print(f"  余弦相似度比例: {sim_ratio:.4f} (越接近1越好)")


        return scores

    def save_scores(self, scores, output_dir):
        """保存评估结果到CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df = pd.DataFrame.from_dict(scores, orient='index')
        output_path = os.path.join(output_dir, f"{self.model_name}_baseline_comparison.csv")
        df.to_csv(output_path)
        print(f"基线对比分数已保存到 {output_path}")


def EvaluateWithBaseline(model_name):
    """评估生成模型并与真实数据基线进行对比"""
    # 动态获取攻击名称
    generated_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
    real_dir = os.path.join(DATASET_SPLITS_DIR, "processed")

    attack_names = [
        f.split('.')[0] for f in os.listdir(generated_dir)
        if f.endswith('.csv') and os.path.exists(os.path.join(real_dir, f"{f.split('.')[0]}.csv"))
    ]

    if not attack_names:
        print("未找到匹配的攻击类型文件")
        return

    gs = GenerativeScores(model_name, attack_names)
    scores = gs.calculate_scores(generated_dir, real_dir)
    gs.save_scores(scores, os.path.join(OUTPUT_DIR, "baseline_scores"))

    # 输出总体评估结果
    print("\n===== 总体评估结果 =====")
    for attack_name, metrics in scores.items():
        print(f"\n[{attack_name}]")
        print(f"  余弦相似度比例: {metrics['cosine_similarity_ratio']:.4f}")




if __name__ == "__main__":
    model_name = "AE-WGAN-AER"  # 模型名称
    EvaluateWithBaseline(model_name)