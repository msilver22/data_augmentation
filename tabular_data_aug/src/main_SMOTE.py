import pandas as pd
import torch
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import ConfusionMatrix
from data_loader import CustomDataModule  
from models.mlp_classifier import MLPClassifier 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants 
DATASET_PATH = 'path_to_your_dataset.csv'  # Dataset path
NUMERIC_COLS = ['HB']  # List of numeric columns in the dataset
CATEGORICAL_COLS = ['60+', 'ASA', 'PEPSINOGENO 1 <30', 'AB PC', 'AB TPO', 'A-FLOGOSI', 'TX ERAD']  # List of categorical columns in the dataset
TARGET_COL = 'LESIONI FOLLOW UP'  # Target column name
TEST_SIZE = 0.2  # Proportion of the dataset to include in the test split
N_RUNS = 10  # Number of training runs
AUGMENTATION_SIZE = {0:500 , 1:350} #Example of augmentation size for binary classification
RANDOM_SEED = 22

def main():
    # Load dataset
    print("[LOG] Preprocessing the dataset")
    df = pd.read_csv(DATASET_PATH)

    # Handling numeric columns (replace commas in numbers if necessary)
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

    # Train/test split
    X_train_real, X_test, y_train_real, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # Oversampling the minority class using SMOTENC (handle categorical/continuous feature)
    smote = SMOTENC(categorical_features=[features.index(col) for col in CATEGORICAL_COLS], sampling_strategy=AUGMENTATION_SIZE, k_neighbors=3, random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_real, y_train_real)

    # Data normalization
    scaler = MinMaxScaler()
    columns_toscale = [col for col in NUMERIC_COLS if col in X_train_resampled.columns]
    X_train_resampled[columns_toscale] = scaler.fit_transform(X_train_resampled[columns_toscale])
    X_test[columns_toscale] = scaler.transform(X_test[columns_toscale])  # Use same scaler for test data

    # Train and test sets
    train_set = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    # Initialize DataModule
    data_module = CustomDataModule(train_set, test_set, features, TARGET_COL, batch_size=32)

    # Train and evaluate model
    model = MLPClassifier(input_dim=len(features), output_dim=1)  # Adjust output_dim if multiclass
    accuracy_avg = 0
    f1score_avg = 0
    confmat_avg = torch.zeros([2, 2]).to(device)

    for i in range(N_RUNS):
        print(f"------ Training iteration {i + 1} -------")
        trainer = pl.Trainer(max_epochs=200)
        trainer.fit(model, data_module)

        # Prediction
        predictions = trainer.predict(model, data_module)
        predictions = torch.cat(predictions).to(device)
        y_test_tensor = torch.Tensor(y_test.values).to(device)

        # Calculate accuracy
        accuracy_meter = torchmetrics.Accuracy(task="binary", num_classes=2).to(device)
        accuracy = accuracy_meter(predictions, y_test_tensor)
        accuracy_avg += accuracy
        print(f"Accuracy: {accuracy.item()}")

        # Calculate F1 score
        f1_score_meter = torchmetrics.F1Score(num_classes=2, average="macro", task="binary").to(device)
        f1_score = f1_score_meter(predictions, y_test_tensor)
        f1score_avg += f1_score
        print(f"F1 Score: {f1_score.item()}")

        # Confusion matrix
        confmat = ConfusionMatrix(task="binary", num_classes=2).to(device)
        conf_matrix = confmat(predictions, y_test_tensor)
        confmat_avg += conf_matrix
        print(f"Confusion Matrix: \n{conf_matrix}")

        # Empty cache to avoid memory issues
        torch.cuda.empty_cache()

    print(f'[LOG] Average Accuracy: {accuracy_avg / N_RUNS:.2f}')
    print(f'[LOG] Average F1 Score: {f1score_avg / N_RUNS:.2f}')
    print('[LOG] Confusion matrix:')
    print(confmat_avg / N_RUNS)
    print(torch.round(confmat_avg / N_RUNS))

if __name__ == "__main__":
    main()
