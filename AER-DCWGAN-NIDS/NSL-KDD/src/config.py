# config.py
import os


RAW_DATA_DIR = os.path.join(".", "raw")
PREPROCESSED_DATA_DIR = os.path.join(".")
DATASET_SPLITS_DIR = os.path.join(".", "Dataset_Splits")
GENERATED_DATA_DIR = os.path.join(".", "generated")
OUTPUT_DIR = os.path.join(".", "output")
SCORES_DIR = os.path.join(OUTPUT_DIR, "scores")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
SCALER_PATH = os.path.join("models", "scaler_pipeline.pkl")
LABEL_MAP_PATH = os.path.join("models", "label_map.pkl")
OUTPUT_PATH = os.path.join(GENERATED_DATA_DIR, "synthetic_dataset.csv")
SAVE_DIR = os.path.join(os.getcwd(), "models")
BATCH_SIZE = 64
EPOCHS = 80


LATENT_DIM_MIN = 16
LATENT_DIM_MAX = 64
RECON_WEIGHT_MIN = 1.0
RECON_WEIGHT_MAX = 5.0
GP_WEIGHT_MIN = 3.0
GP_WEIGHT_MAX = 10.0
AER_WEIGHT_MIN = 0.5
AER_WEIGHT_MAX = 3.0


ENCODER_HIDDEN_DIM = 20
GENERATOR_HIDDEN_DIMS = [20, 40, 80, 160]
DISCRIMINATOR_HIDDEN_DIMS = [120, 80, 40]
ENCODER_DISCRIMINATOR_HIDDEN_DIMS = [120, 80, 40]


LEARNING_RATE = 1e-5
GRADIENT_CLIP = 2.0
GRADIENT_PENALTY_COEFFICIENT = 10.0