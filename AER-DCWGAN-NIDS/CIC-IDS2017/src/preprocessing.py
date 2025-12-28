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


plt.rcParams["axes.unicode_minus"] = False


def preprocess_cicids2017(raw_data_dir, dataset_split_dir):

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

        df = pd.read_csv(
            os.path.join(raw_data_dir, f),
            header=0,
            engine='c',
            low_memory=False
        )


        df.columns = [col.strip().replace(' ', '_') for col in df.columns]


        required_cols = [col for col in df.columns if col != 'Label']


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

    def map_attack_labels(label):
        label = label.strip()

        if label.startswith("Web Attack � "):
            return "Web Attack"


        if "SSH-Patator" in label:
            return "Patator"
        if "FTP-Patator" in label:
            return "Patator"

        return label

    joblib.dump(label_map, LABEL_MAP_PATH)



    data['Label'] = data['Label'].astype(str).apply(map_attack_labels)


    unknown_labels = set(data['Label'].unique()) - set(label_map.keys())
    if unknown_labels:

        data = data[data['Label'].isin(label_map.keys())]
    def clean_data(df):

        df.replace([np.inf, -np.inf], np.nan, inplace=True)


        df = df.dropna(axis=1, how='all')

        feature_cols = [col for col in df.columns if col != 'Label']
        for col in feature_cols:
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)

        df = df.dropna(subset=['Label'])

        return df

    data = clean_data(data)

    for col in data.columns:
        missing_count = data[col].isnull().sum()


    print(f"\n原始数据集大小: {len(data)}")


    train, temp = train_test_split(
        data,
        test_size=0.4,
        stratify=data['Label'],
        random_state=42
    )


    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp['Label'],
        random_state=42
    )


    X_train = train.drop(columns=['Label']).values
    y_train_str = train['Label'].values
    X_val = val.drop(columns=['Label']).values
    y_val_str = val['Label'].values
    X_test = test.drop(columns=['Label']).values
    y_test_str = test['Label'].values

    def create_feature_pipeline():
        return Pipeline([
            ('variance', VarianceThreshold(threshold=0.05)),
            ('rfe', RFE(
                estimator=LogisticRegression(max_iter=1000, multi_class='multinomial'),
                n_features_to_select=30
            )),
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])

    pipeline = create_feature_pipeline()
    pipeline.fit(X_train, y_train_str)
    joblib.dump(pipeline, SCALER_PATH)


    visualize_feature_importance(pipeline, data.drop(columns=['Label']).columns)

    def save_processed_data(X, y, file_name):

        processed_df = pd.DataFrame(X)
        processed_df['Label'] = y


        processed_df.to_csv(os.path.join(dataset_split_dir, file_name), index=False)


    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)


    save_processed_data(X_train_processed, y_train_str, "train_processed.csv")
    save_processed_data(X_val_processed, y_val_str, "val_processed.csv")
    save_processed_data(X_test_processed, y_test_str, "test_processed.csv")


    X_full = data.drop(columns=['Label']).values
    y_full_str = data['Label'].values
    X_full_processed = pipeline.transform(X_full)
    save_processed_data(X_full_processed, y_full_str, "full_processed.csv")


def visualize_feature_importance(pipeline, original_feature_names):

    try:

        rfe = pipeline.named_steps['rfe']

        selected_features_mask = rfe.support_

        all_features = np.array([f"feature_{i}" for i in range(len(original_feature_names))])


        selected_features = all_features[selected_features_mask]

        feature_importance = rfe.estimator_.coef_

        if len(feature_importance.shape) > 1:
            feature_importance = np.mean(np.abs(feature_importance), axis=0)

        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)


        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])

        plt.tight_layout()


        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')



        plt.show()



        for i, (feature, importance) in enumerate(importance_df.values):
            print(f"{i + 1}. {feature}: {importance:.4f}")

    except Exception as e:
        print(f" {e}")


def split_dataset(dataset_splits_dir):

    label_map = joblib.load(LABEL_MAP_PATH)
    reverse_label_map = {v: k for k, v in label_map.items()}

    splits = [
        {"source": "full_processed.csv", "dest_dir": "processed", "prefix": ""},  #
    ]

    for split_info in splits:
        src_file = os.path.join(dataset_splits_dir, split_info["source"])
        dest_dir = os.path.join(dataset_splits_dir, split_info["dest_dir"])
        prefix = split_info["prefix"]
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(src_file):

            continue

        df = pd.read_csv(src_file)
        _split_and_save(df, dest_dir, prefix)


def _split_and_save(df, dest_dir, prefix):

    for label in df['Label'].unique():
        safe_name = label.replace(' ', '_').replace('/', '-')
        dest_path = os.path.join(dest_dir, f"{safe_name}.csv" if prefix == "" else f"{prefix}{safe_name}.csv")

        subset = df[df['Label'] == label]
        if not subset.empty:
            subset.to_csv(dest_path, index=False)




def inverse_scale_data(data):

    pipeline = joblib.load(SCALER_PATH)
    return pipeline.named_steps['scaler'].inverse_transform(data)


if __name__ == "__main__":
    try:
        preprocess_cicids2017(RAW_DATA_DIR, DATASET_SPLITS_DIR)
        split_dataset(DATASET_SPLITS_DIR)
    except Exception as e:

        raise