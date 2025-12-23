import numpy as np
import pandas as pd
import os
from config import  DATASET_SPLITS_DIR, GENERATED_DATA_DIR, OUTPUT_DIR, LABEL_MAP_PATH
import joblib

def merge_datasets():
    print("\n开始合并数据集...")
    # 定义合并后的数据保存目录
    merged_data_dir = os.path.join(OUTPUT_DIR, "merged_data")
    os.makedirs(merged_data_dir, exist_ok=True)

    # 加载预处理后的训练集
    train_path = os.path.join(DATASET_SPLITS_DIR, "train_processed.csv")  # 修改：指向实际的训练CSV文件

    # 检查训练文件是否存在
    if not os.path.exists(train_path):
        print(f"错误：找不到训练文件 {train_path}")
        print("请确保预处理步骤正确执行并生成了train_processed.csv文件")
    else:
        # 读取训练数据
        train_df = pd.read_csv(train_path)

        all_generated_dfs = []  # 用于存储所有生成的数据集

        # 获取所有攻击类型（基于生成的数据）
        generated_dir = os.path.join(GENERATED_DATA_DIR, "generated_attacks")
        generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.csv')]

        for file in generated_files:
            generated_path = os.path.join(generated_dir, file)
            # 加载生成的数据
            generated_df = pd.read_csv(generated_path)
            all_generated_dfs.append(generated_df)  # 将生成的数据集添加到列表中

        # 合并所有生成的数据集
        total_generated_df = pd.concat(all_generated_dfs, axis=0, ignore_index=True)

        # 合并训练集和所有生成的数据集
        total_merged_df = pd.concat([train_df, total_generated_df], axis=0, ignore_index=True)

        # 保存总的数据集
        total_merged_path = os.path.join(OUTPUT_DIR, "total_merged_data.csv")
        total_merged_df.to_csv(total_merged_path, index=False)
        print(f"已保存总的合并后数据集: {total_merged_path} (样本数: {len(total_merged_df)})")
if __name__ == "__main__":
    merge_datasets()