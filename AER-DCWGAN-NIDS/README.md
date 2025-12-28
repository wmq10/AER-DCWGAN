AER-DCWGAN: Network Intrusion Detection with Adversarial Encoder Regularization
Official PyTorch implementation of the paper "AER-DCWGAN for Data Augmentation in Network Intrusion Detection" (Submitted to Scientific Reports).

Quick Start
1. Clone and Install Dependencies
bash
https://github.com/wmq10/AER-DCWGAN/AER-DCWGAN-NIDS.git
cd AER-DCWGAN-NIDS
pip install -r requirements.txt
2. Download Datasets
CIC-IDS2017: Download from the official site. Place the extracted CSV files in the data/raw/ directory.

NSL-KDD: Available from the Canadian Institute for Cybersecurity.

3. Run the Pipeline
Execute the following commands in order:

bash
# 1. Preprocess the data
python src/preprocessing.py

# 2. Train the model and generate augmented samples
python src/train.py

# 3. Evaluate the quality of generated samples
python src/evaluate.py
Project Structure

Key Features
The AER-DCWGAN model integrates three main components to address data imbalance in intrusion detection:

Dual-Conditional Embedding Generator for targeted sample generation.

Label-Aware Gradient Penalty to enhance discriminator sensitivity.

Adversarial Encoder Regularization (AER) to improve feature diversity and training stability.

Reproducing Results
To replicate the key results from the paper using the optimal hyperparameters:

bash
python src/train.py
Training logs, generated samples, and evaluation metrics will be saved in the outputs/ directory.

Data and Code Availability
Datasets: The public datasets used (CIC-IDS2017, NSL-KDD) are available from the sources linked above.

Code: This repository contains the complete code for preprocessing, model training, and evaluation.