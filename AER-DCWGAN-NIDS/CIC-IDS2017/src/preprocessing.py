import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from config import RAW_DATA_DIR, DATASET_SPLITS_DIR, SCALER_PATH, LABEL_MAP_PATH, OUTPUT_PATH

# 设置中文字体
plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def preprocess_cicids2017(raw_data_dir, dataset_split_dir):
    # 加载和合并数据
    files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv"
    ]

    dfs = []
    for f in files:
        # 读取原始数据
        df = pd.read_csv(
            os.path.join(raw_data_dir, f),
            header=0,
            engine='c',
            low_memory=False  # 防止内存溢出
        )

        # 清洗列名：去除空格和特殊字符
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]

        # 验证列结构
        required_cols = [col for col in df.columns if col != 'Label']

        # 重命名特征列为统一格式
        df.columns = [f'feature_{i}' for i in range(78)] + ['Label']

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    label_map = {
        'BENIGN': 0, 'Bot': 1, 'PortScan': 2, 'DDoS': 3,
        'Infiltration': 4, 'Heartbleed': 5, 'Patator': 6,
        'DoS Slowhttptest': 7, 'DoS Hulk': 8, 'DoS slowloris': 9,
        'DoS GoldenEye': 10, 'Web Attack': 11
    }
    reverse_label_map = {v: k for k, v in label_map.items()}

    # 新增：处理Web攻击标签的函数
    def map_attack_labels(label):
        label = label.strip()

        # 处理Web攻击
        if label.startswith("Web Attack � "):
            return "Web Attack"

        # 处理Patator攻击（合并FTP/SSH-Patator）
        if "SSH-Patator" in label:
            return "Patator"
        if "FTP-Patator" in label:
            return "Patator"

        return label

    joblib.dump(label_map, LABEL_MAP_PATH)
    print(f"标签映射表已保存至: {LABEL_MAP_PATH}")

    # 应用标签清理
    data['Label'] = data['Label'].astype(str).apply(map_attack_labels)

    # 检查未知标签
    unknown_labels = set(data['Label'].unique()) - set(label_map.keys())
    if unknown_labels:
        print(f"发现未知标签: {unknown_labels}")
        data = data[data['Label'].isin(label_map.keys())]
        print(f"移除未知标签后，剩余样本: {len(data)}")

    # 数据清洗 - 使用均值填充NaN
    def clean_data(df):
        # 处理无穷大值
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 删除全为NA的列
        df = df.dropna(axis=1, how='all')

        # 使用均值填充特征列中的NaN
        feature_cols = [col for col in df.columns if col != 'Label']
        for col in feature_cols:
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)

        # 确保标签列无缺失（二次检查）
        df = df.dropna(subset=['Label'])

        return df

    # 应用清洗流程
    data = clean_data(data)

    # 打印缺失值统计信息
    print("\n缺失值统计：")
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            print(f"{col}: {missing_count} 个缺失值 (已用均值填充)")

    # 数据集划分 - 按6:2:2划分训练集、验证集和测试集
    print(f"\n原始数据集大小: {len(data)}")

    # 首先将数据划分为训练集(60%)和临时集(40%)
    train, temp = train_test_split(
        data,
        test_size=0.4,  # 40%数据用于验证集和测试集
        stratify=data['Label'],
        random_state=42
    )

    # 将临时集平均分为验证集和测试集，各占20%
    val, test = train_test_split(
        temp,
        test_size=0.5,  # 各占原始数据的20%
        stratify=temp['Label'],
        random_state=42
    )

    # 打印数据集划分结果
    print(f"训练集大小: {len(train)} ({len(train) / len(data):.1%})")
    print(f"验证集大小: {len(val)} ({len(val) / len(data):.1%})")
    print(f"测试集大小: {len(test)} ({len(test) / len(data):.1%})")

    # 提取特征和标签
    X_train = train.drop(columns=['Label']).values
    y_train_str = train['Label'].values  # 保留字符标签用于保存
    X_val = val.drop(columns=['Label']).values
    y_val_str = val['Label'].values
    X_test = test.drop(columns=['Label']).values
    y_test_str = test['Label'].values

    # 定义特征工程pipeline
    def create_feature_pipeline():
        return Pipeline([
            ('variance', VarianceThreshold(threshold=0.05)),  # 过滤方差<0.05的特征
            ('rfe', RFE(  # 递归特征选择（RFE）
                estimator=LogisticRegression(max_iter=1000, multi_class='multinomial'),
                n_features_to_select=30 # 选择24个特征
            )),
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))  # 归一化
        ])

    # 训练集：拟合pipeline并保存
    pipeline = create_feature_pipeline()
    pipeline.fit(X_train, y_train_str)
    joblib.dump(pipeline, SCALER_PATH)
    print(f"特征工程pipeline已保存至: {SCALER_PATH}")

    # 可视化特征重要性
    visualize_feature_importance(pipeline, data.drop(columns=['Label']).columns)

    # 保存预处理后的训练集、验证集和测试集
    def save_processed_data(X, y, file_name):
        # 将特征和标签组合成DataFrame
        processed_df = pd.DataFrame(X)
        processed_df['Label'] = y  # 使用字符标签

        # 保存为CSV文件
        processed_df.to_csv(os.path.join(dataset_split_dir, file_name), index=False)
        print(f"已保存处理后的数据集: {file_name} (特征数: {X.shape[1]}, 样本数: {len(y)})")

    # 应用特征工程并保存
    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    # 保存划分后的train/val/test，使用字符标签
    save_processed_data(X_train_processed, y_train_str, "train_processed.csv")
    save_processed_data(X_val_processed, y_val_str, "val_processed.csv")
    save_processed_data(X_test_processed, y_test_str, "test_processed.csv")

    # 保存完整处理后的数据集
    X_full = data.drop(columns=['Label']).values
    y_full_str = data['Label'].values
    X_full_processed = pipeline.transform(X_full)
    save_processed_data(X_full_processed, y_full_str, "full_processed.csv")


def visualize_feature_importance(pipeline, original_feature_names):
    """可视化特征重要性"""
    try:
        # 获取特征选择器
        rfe = pipeline.named_steps['rfe']

        # 获取被选中的特征掩码
        selected_features_mask = rfe.support_

        # 获取原始特征名称
        all_features = np.array([f"feature_{i}" for i in range(len(original_feature_names))])

        # 获取选中的特征名称
        selected_features = all_features[selected_features_mask]

        # 获取特征重要性得分
        feature_importance = rfe.estimator_.coef_

        # 如果是多分类问题，计算每个特征在所有类别上的平均重要性
        if len(feature_importance.shape) > 1:
            feature_importance = np.mean(np.abs(feature_importance), axis=0)

        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title('特征重要性分析')
        plt.tight_layout()

        # 保存图表
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {OUTPUT_PATH}")

        # 显示图表
        plt.show()

        # 打印特征重要性排名
        print("\n特征重要性排名:")
        for i, (feature, importance) in enumerate(importance_df.values):
            print(f"{i + 1}. {feature}: {importance:.4f}")

    except Exception as e:
        print(f"可视化特征重要性时出错: {e}")


def split_dataset(dataset_splits_dir):
    """按攻击类型拆分数据集，完整数据集文件名无任何前缀"""
    label_map = joblib.load(LABEL_MAP_PATH)
    reverse_label_map = {v: k for k, v in label_map.items()}

    # 定义数据集拆分配置（完整数据集无前缀，其他带前缀）
    splits = [
        {"source": "full_processed.csv", "dest_dir": "processed", "prefix": ""},  # 完整数据集
    ]

    for split_info in splits:
        src_file = os.path.join(dataset_splits_dir, split_info["source"])
        dest_dir = os.path.join(dataset_splits_dir, split_info["dest_dir"])
        prefix = split_info["prefix"]
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(src_file):
            print(f"跳过 {split_info['source']}: 源文件不存在")
            continue

        df = pd.read_csv(src_file)
        _split_and_save(df, dest_dir, prefix)


def _split_and_save(df, dest_dir, prefix):
    """辅助函数：按标签拆分，完整数据集文件名无前缀"""
    for label in df['Label'].unique():
        safe_name = label.replace(' ', '_').replace('/', '-')
        dest_path = os.path.join(dest_dir, f"{safe_name}.csv" if prefix == "" else f"{prefix}{safe_name}.csv")

        subset = df[df['Label'] == label]
        if not subset.empty:
            subset.to_csv(dest_path, index=False)
            print(f"已保存: {dest_path} ({len(subset)}条)")
        else:
            print(f"警告: {'完整数据集' if prefix == '' else f'{prefix[:-1]}数据集'}中无{label}数据")


def inverse_scale_data(data):
    """逆归一化"""
    pipeline = joblib.load(SCALER_PATH)
    return pipeline.named_steps['scaler'].inverse_transform(data)


if __name__ == "__main__":
    try:
        preprocess_cicids2017(RAW_DATA_DIR, DATASET_SPLITS_DIR)
        split_dataset(DATASET_SPLITS_DIR)
    except Exception as e:
        print(f"预处理失败: {str(e)}")
        raise