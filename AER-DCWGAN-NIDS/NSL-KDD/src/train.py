# main.py 文件
import torch
import time
import numpy as np
import pandas as pd
import warnings
from preprocessing import preprocess_nslkdd, split_dataset
from models import AE_WGAN_AER
from generative_scores import GenerativeScores
import os
from config import BATCH_SIZE, EPOCHS
from config import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, DATASET_SPLITS_DIR, GENERATED_DATA_DIR, SCALER_PATH, LABEL_MAP_PATH, OUTPUT_DIR
import joblib
from merge_data import merge_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def main():
    # 初始化计时器
    start_time = time.time()

    # main.py 中合并预处理检查
    if not os.path.exists(PREPROCESSED_DATA_DIR) or not os.path.exists(SCALER_PATH):
        print("正在执行数据预处理...")
        preprocess_nslkdd(RAW_DATA_DIR, DATASET_SPLITS_DIR)  # 修改：调用NSL - KDD预处理函数
        split_dataset(DATASET_SPLITS_DIR)
    else:
        print("预处理数据及scaler已存在，跳过预处理步骤。")

    # 加载全局scaler和标签映射
    try:
        scaler = joblib.load(SCALER_PATH)
        label_map = joblib.load(LABEL_MAP_PATH)
        reverse_label_map = {v: k for k, v in label_map.items()}  # 创建逆向映射
        print(f"成功加载Scaler文件: {SCALER_PATH}")
        print(f"成功加载标签映射文件: {LABEL_MAP_PATH}")
    except Exception as e:
        print(f"加载文件失败: {e}")
        print("正在重新运行预处理步骤...")
        preprocess_nslkdd(RAW_DATA_DIR, DATASET_SPLITS_DIR)
        split_dataset(DATASET_SPLITS_DIR)
        scaler = joblib.load(SCALER_PATH)
        label_map = joblib.load(LABEL_MAP_PATH)
        reverse_label_map = {v: k for k, v in label_map.items()}  # 创建逆向映射

    # --- 模型训练阶段 ---
    print("\n进入模型训练阶段:")
    train_dir = os.path.join(DATASET_SPLITS_DIR, "processed")
    all_attacks = [f.split('.')[0] for f in os.listdir(train_dir) if f.endswith('.csv')]

    # 加载预处理后的数据并验证特征数量
    for attack_idx, attack_name in enumerate(all_attacks):
        epoch_start = time.time()
        print(f"\n处理攻击类别 ({attack_idx + 1}/{len(all_attacks)}): {attack_name}")

        # 加载数据
        attack_path = os.path.join(train_dir, f"{attack_name}.csv")
        attack_data = pd.read_csv(attack_path)

        # 修改：根据实际预处理后的数据列数调整
        if attack_data.shape[1] != 25:  # 24特征 + 1标签
            raise ValueError(f"数据维度错误: {attack_name} 包含 {attack_data.shape[1]} 列，应为25（24特征+1标签）")

        X_attack = attack_data.drop('Label', axis=1).values.astype(np.float32)
        X_scaled = X_attack

        # 重塑为模型输入形状
        try:
            reshaped_X = X_scaled.reshape(-1, 1, 4, 6)  # 调整为合适的形状
        except ValueError as e:
            print(f"形状转换错误: {X_scaled.shape} -> (-1,1,4,6)")
            raise
        reshaped_tensor = torch.from_numpy(reshaped_X).to(device)
        # 获取现有样本数量
        existing_samples = len(attack_data)

        # 根据现有样本数量动态确定需要生成的样本数量
        if existing_samples < 10000:
            num_to_generate = 30000
        elif 10000 <= existing_samples < 20000:
            num_to_generate = 20000
        elif 20000 <= existing_samples < 30000:
            num_to_generate = 10000
        else:
            num_to_generate = 2000

        # 获取标签
        labels = attack_data['Label'].values
        # 将标签转换为数值类型
        labels = np.array([label_map[label] for label in labels])
        # --- 仅在需要生成数据时训练模型 ---
        if num_to_generate > 0:
            print(f"正在生成 {num_to_generate} 条样本（现有样本：{existing_samples}）")

            # 初始化模型，传递正确的类别数
            ae_wgan = AE_WGAN_AER(
                img_shape=(1, 4, 6),
                num_classes=5,  # 类别数为5
                latent_dim=32
            ).to(device)

            # 模型训练
            training_log = ae_wgan.train(
                attack_name=attack_name,
                reshaped_X_attack=reshaped_tensor,
                labels=labels,  # 传递标签
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )

            # 生成数据
            generated_samples = ae_wgan.generate_data(num_samples=num_to_generate)
            generated_denorm = generated_samples.reshape(-1, 24)  # 调整为24个特征
        else:
            print(f"跳过训练: {attack_name} 样本已充足，现有样本数 {existing_samples}")
            generated_denorm = pd.DataFrame()  # 创建空DataFrame

            # ============= 保存逻辑 =============
        save_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{attack_name}.csv")

        # 仅当有生成数据时保存
        if generated_denorm.size != 0:  # 检查 NumPy 数组是否为空
            generated_df = pd.DataFrame(generated_denorm)
            generated_df['Label'] = attack_name  # 使用攻击名称作为标签
            generated_df.to_csv(save_path, index=False)
            del generated_df  # 安全删除
        else:
            print(f"未生成新样本，跳过保存: {attack_name}")

        print(f"完成 {attack_name} 生成，耗时 {(time.time() - epoch_start) / 60:.1f} 分钟")

    print("\n开始生成质量评估:")
    # 使用 GenerativeScores 类进行评估（替换原有的未定义函数）
    gs = GenerativeScores(model_name='AE-WGAN-AER', attack_names=all_attacks)
    scores = gs.calculate_scores(
        generated_dir=os.path.join(GENERATED_DATA_DIR, "generated_attacks"),
        real_dir=os.path.join(DATASET_SPLITS_DIR, "processed")
    )
    gs.save_scores(scores, output_dir=os.path.join(OUTPUT_DIR, "baseline_scores"))

    # 调用合并数据的函数
    merge_datasets()

    # --- 性能统计 ---
    total_time = (time.time() - start_time) / 60
    print(f"\n总耗时: {total_time:.1f} 分钟")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()