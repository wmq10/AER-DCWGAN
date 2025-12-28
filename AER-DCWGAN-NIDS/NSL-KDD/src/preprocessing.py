import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.impute import SimpleImputer
import joblib
from config import RAW_DATA_DIR, DATASET_SPLITS_DIR, SCALER_PATH, LABEL_MAP_PATH
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_nslkdd(raw_data_dir, dataset_split_dir):

    train_file = "KDDTrain+.txt"
    test_file = "KDDTest+.txt"
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]

    train_df = pd.read_csv(
        os.path.join(raw_data_dir, train_file),
        header=None,
        names=column_names,
        engine='c',
        low_memory=False
    )

    test_df = pd.read_csv(
        os.path.join(raw_data_dir, test_file),
        header=None,
        names=column_names,
        engine='c',
        low_memory=False
    )


    categorical_columns = ['protocol_type', 'service', 'flag']
    train_df = pd.get_dummies(train_df, columns=categorical_columns)
    test_df = pd.get_dummies(test_df, columns=categorical_columns)


    all_columns = set(train_df.columns).union(set(test_df.columns))
    for df in [train_df, test_df]:
        for col in all_columns:
            if col not in df.columns:
                df[col] = 0
        df.sort_index(axis=1, inplace=True)


    attack_type_mapping = {
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
        'spy': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
        'normal': 'normal'
    }
    train_df['attack_type'] = train_df['attack_type'].map(attack_type_mapping)
    test_df['attack_type'] = test_df['attack_type'].map(attack_type_mapping)

    #
    unique_labels = sorted(train_df['attack_type'].unique())
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    joblib.dump(label_to_num, LABEL_MAP_PATH)  #



    def clean_data(df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(axis=1, how='all')
        feature_cols = [col for col in df.columns if col != 'attack_type']

        for col in feature_cols:
            if df[col].isna().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
        df = df.dropna(subset=['attack_type'])
        return df

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['attack_type'], random_state=42
    )


    X_train = train_df.drop(columns=['attack_type']).values
    y_train = train_df['attack_type'].values
    X_val = val_df.drop(columns=['attack_type']).values
    y_val = val_df['attack_type'].values
    X_test = test_df.drop(columns=['attack_type']).values
    y_test = test_df['attack_type'].values


    def create_feature_pipeline():
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  #
            ('variance', VarianceThreshold(threshold=0.05)),  #
            ('rfe', RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=24)),  #
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))  #
        ])

    # -
    pipeline = create_feature_pipeline()
    pipeline.fit(X_train, y_train)

    #
    minmax_scaler = pipeline.named_steps['scaler']
    joblib.dump(minmax_scaler, SCALER_PATH)  #



    variance_selector = pipeline.named_steps['variance']
    original_features = train_df.drop(columns=['attack_type']).columns
    kept_indices = variance_selector.get_support(indices=True)
    kept_features = original_features[kept_indices].tolist()



    rfe_selector = pipeline.named_steps['rfe']
    rfe_indices = rfe_selector.get_support(indices=True)
    rfe_features = np.array(kept_features)[rfe_indices].tolist()
    rfe_importances = rfe_selector.estimator_.feature_importances_


    feature_importance = pd.DataFrame({
        'Feature': rfe_features,
        'Importance': rfe_importances
    }).sort_values('Importance', ascending=False)
    print(feature_importance.head(10))


    X_train_rfe = pipeline.transform(X_train)
    X_test_rfe = pipeline.transform(X_test)


    base_model = RandomForestClassifier(random_state=42)
    base_model.fit(X_train_rfe, y_train)
    y_pred_base = base_model.predict(X_test_rfe)
    base_accuracy = accuracy_score(y_test, y_pred_base)


    def save_processed_data(X, y, file_name):
        processed_df = pd.DataFrame(X)
        processed_df['Label'] = y
        processed_df.to_csv(os.path.join(dataset_split_dir, file_name), index=False)


    X_train_processed = pipeline.transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)
    X_full = np.vstack((X_train, X_val, X_test))
    y_full = np.hstack((y_train, y_val, y_test))
    X_full_processed = pipeline.transform(X_full)

    save_processed_data(X_full_processed, y_full, "full_processed.csv")
    save_processed_data(X_train_processed, y_train, "train_processed.csv")
    save_processed_data(X_val_processed, y_val, "val_processed.csv")
    save_processed_data(X_test_processed, y_test, "test_processed.csv")



def split_dataset(dataset_splits_dir):
    splits = [{"source": "full_processed.csv", "dest_dir": "processed", "prefix": ""}]
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
    for label_name in df['Label'].unique():
        safe_name = label_name.replace(' ', '_').replace('/', '-')
        dest_path = os.path.join(dest_dir, f"{prefix}{safe_name}.csv")
        subset = df[df['Label'] == label_name]



if __name__ == "__main__":
    try:
        preprocess_nslkdd(RAW_DATA_DIR, DATASET_SPLITS_DIR)
        split_dataset(DATASET_SPLITS_DIR)
    except Exception as e:

        raise