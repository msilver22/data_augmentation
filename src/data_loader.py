import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import MNIST
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


"""-----MNIST Data Module-----"""

class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)



"""-----Fashion MNIST Data Module-----"""


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dims = (1, 28, 28)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            fmnist_full = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.fmnist_train, self.fmnist_val = random_split(fmnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.fmnist_test = datasets.FashionMNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.fmnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.fmnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.fmnist_test, batch_size=self.batch_size, num_workers=self.num_workers)



"-----Tabular-data Data Module-----"

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

