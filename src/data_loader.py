import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

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
