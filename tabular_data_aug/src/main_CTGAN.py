import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
from sdv.tabular import CTGANSynthesizer
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import ConfusionMatrix
from data_loader import CustomDataModule
from models.mlp_classifier import MLPClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants
DATASET_PATH = 'path_to_your_dataset.csv'  # Dataset path
NUMERIC_COLS = ['HB']  # List of numeric columns in the dataset
CATEGORICAL_COLS = ['60+', 'ASA', 'PEPSINOGENO 1 <30', 'AB PC', 'AB TPO', 'A-FLOGOSI', 'TX ERAD']  # Categorical columns
TARGET_COL = 'LESIONI FOLLOW UP'  # Target column
TEST_SIZE = 0.2  # Proportion of the dataset to use as test set
N_RUNS = 10  # Number of training runs
N_SYNTHETIC_SAMPLES = 161  # Number of synthetic samples to generate for the minority class

def main():
    # Load dataset
    print("[LOG] Preprocessing the dataset")
    df = pd.read_csv(DATASET_PATH)

    # Handle numeric columns (replace commas in numbers if necessary)
    for col in NUMERIC_COLS:
        df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype(float)

    # Data imputation
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numeric_imputer = SimpleImputer(strategy='median')
    df[CATEGORICAL_COLS] = categorical_imputer.fit_transform(df[CATEGORICAL_COLS])
    df[NUMERIC_COLS] = numeric_imputer.fit_transform(df[NUMERIC_COLS])
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(int)
    print("[LOG] Preprocessing completed")

    # Classification Setup
    features = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[features]
    y = df[TARGET_COL]

    # Split the data into training and test sets
    X_train_real, X_test, y_train_real, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    # Identify the minority class (assuming binary classification)
    class_counts = y_train_real.value_counts()
    minority_class = class_counts.idxmin()
    print(f"[LOG] Minority class identified: {minority_class}")

    # Select rows corresponding to the minority class for augmentation
    X_minority = X_train_real[y_train_real == minority_class]
    
    # Augment the minority class using CTGAN
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_minority)
    synthesizer = CTGANSynthesizer(
        metadata, 
        epochs=2000,
        verbose=False,
        embedding_dim=256,
        generator_dim=(512, 512),
        discriminator_dim=(512, 512),
        discriminator_steps=2,
        pac=1
    )
    synthesizer.fit(X_minority)

    accuracy_avg = 0
    f1score_avg = 0
    confmat_avg = torch.zeros([2, 2]).to(device)

    for i in range(N_RUNS):
        print(f"------ Training iteration {i + 1} -------")

        # Generate synthetic data for the minority class
        synthetic_data = synthesizer.sample(num_rows=N_SYNTHETIC_SAMPLES)
        
        # Augment training data by concatenating real training data with synthetic data
        X_train_augmented = pd.concat([X_train_real, synthetic_data], ignore_index=True)
        synthetic_labels = np.repeat(minority_class, N_SYNTHETIC_SAMPLES)
        y_train_augmented = pd.concat([y_train_real, pd.Series(synthetic_labels, name=TARGET_COL)], ignore_index=True)

        # Data normalization
        scaler = MinMaxScaler()
        columns_to_scale = ['A-FLOGOSI', 'HB']
        X_train_augmented[columns_to_scale] = scaler.fit_transform(X_train_augmented[columns_to_scale])
        X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        # Create training and test sets
        train_set = pd.concat([X_train_augmented, y_train_augmented], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)
        data_module = CustomDataModule(train_set, test_set, features, TARGET_COL, batch_size=32)

        # Model training
        model = MLPClassifier(input_dim=len(features), output_dim=1)
        trainer = pl.Trainer(max_epochs=200)
        trainer.fit(model, data_module)

        # Prediction and evaluation
        predictions = trainer.predict(model, data_module)
        predictions = torch.cat(predictions).to(device)
        y_test_tensor = torch.Tensor(y_test.values).to(device)

        # Calculate accuracy
        accuracy_meter = torchmetrics.Accuracy(task="binary", num_classes=2).to(device)
        accuracy = accuracy_meter(predictions, y_test_tensor)
        accuracy_avg += accuracy

        # Calculate F1 score
        f1_score_meter = torchmetrics.F1Score(num_classes=2, average="macro", task="binary").to(device)
        f1_score = f1_score_meter(predictions, y_test_tensor)
        f1score_avg += f1_score

        # Confusion matrix
        confmat = ConfusionMatrix(task="binary", num_classes=2).to(device)
        conf_matrix = confmat(predictions, y_test_tensor)
        confmat_avg += conf_matrix

        # Clear memory cache to avoid memory issues
        torch.cuda.empty_cache()

    print(f'[LOG] Average Accuracy: {accuracy_avg / N_RUNS:.2f}')
    print(f'[LOG] Average F1 Score: {f1score_avg / N_RUNS:.2f}')
    print('[LOG] Confusion matrix:')
    print(confmat_avg / N_RUNS)
    print(torch.round(confmat_avg / N_RUNS))

if __name__ == "__main__":
    main()
