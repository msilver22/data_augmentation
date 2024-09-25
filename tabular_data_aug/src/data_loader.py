import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, df, features, target):
        self.features = df[features].values
        self.target = df[target].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.target[idx], dtype=torch.long)

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, df, features, target, batch_size=32, test_size=0.2, val_size=0.1, random_state=None):
        super().__init__()
        self.df = df
        self.features = features
        self.target = target
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def setup(self, stage=None):
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size, random_state=self.random_state)

        self.train_dataset = CustomDataset(pd.concat([X_train, y_train], axis=1), self.features, self.target)
        self.val_dataset = CustomDataset(pd.concat([X_val, y_val], axis=1), self.features, self.target)
        self.test_dataset = CustomDataset(pd.concat([X_test, y_test], axis=1), self.features, self.target)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
