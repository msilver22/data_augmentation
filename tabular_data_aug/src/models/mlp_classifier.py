import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim=8, alpha=1):
        super(clf_MLP, self,).__init__()

        self.alpha=alpha

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

        self.alpha = alpha

        self.accuracy_meter = torchmetrics.classification.Accuracy(
            task="binary", num_classes=2
        )
        self.f1_score = torchmetrics.F1Score(num_classes=2, average="macro", task="binary")
        self.epoch_f1_scores = []
        self.y_test=[]

    def forward(self, x):
        return self.model(x).squeeze()


    def training_step(self, batch):
        x, y = batch
        prob = self(x)
        w = torch.where(y == 0, torch.ones_like(y), torch.full_like(y, self.alpha)) 
        loss_fn = nn.BCEWithLogitsLoss(weight=w)
        train_loss = loss_fn(prob, y.float())
        train_accuracy = self.accuracy_meter(prob, y.float())
        values = {"train_loss": train_loss, "train_acc": train_accuracy}
        self.log_dict(values, prog_bar=True)

        return train_loss
    
    def validation_step(self, batch):
        x, y = batch
        prob = self(x)
        w = torch.where(y == 0, torch.ones_like(y), torch.full_like(y, self.alpha)) 
        loss_fn = nn.BCEWithLogitsLoss(weight=w)
        val_loss = loss_fn(prob, y.float())
        val_accuracy = self.accuracy_meter(prob, y.float())
        values = {"val_loss": val_loss, "val_acc": val_accuracy}
        self.log_dict(values, prog_bar=True)
        preds = (prob >= 0.5).int()
        self.f1_score(preds, y.int())

        return val_loss
    
    def on_validation_epoch_end(self):
        f1 = self.f1_score.compute()
        self.log("val_f1", f1, on_epoch=True, prog_bar=True, logger=True)
        self.epoch_f1_scores.append(f1.item())
        self.f1_score.reset()

    def test_step(self, batch):
        x, y = batch
        prob = self(x)
        w = torch.where(y == 0, torch.ones_like(y), torch.full_like(y, self.alpha)) 
        loss_fn = nn.BCEWithLogitsLoss(weight=w)
        test_loss = loss_fn(prob, y.float())
        test_accuracy = self.accuracy_meter(prob, y)
        values = {"test_loss": test_loss, "test_acc": test_accuracy}
        self.log_dict(values, prog_bar=True)
        preds = (prob >= 0.5).int()
        self.f1_score(preds, y.int())
        return test_loss
    
    def on_test_epoch_end(self):
        f1 = self.f1_score.compute()
        self.log("test_f1", f1, prog_bar=True, logger=True)
        self.f1_score.reset()

    def predict_step(self, batch):
        x, y = batch
        self.y_test.append(y)
        prob = self(x)
        preds = (prob >= 0.5).int()

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
