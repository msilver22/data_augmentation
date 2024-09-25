import torch
import torch.nn as nn
import lightning as L
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count() / 2)
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class Encoder(nn.Module):
    def __init__(self, in_features=784, z_dim=2):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.fc = nn.Sequential(
            nn.Linear(in_features+10, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.out1 = nn.Linear(256, z_dim)
        self.out2 = nn.Linear(256, z_dim)

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        h = self.fc(x)
        mu = self.out1(h)
        log_var = self.out2(h)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, in_features=2, out_features=784):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(in_features+10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, out_features),
            nn.Tanh()
        )

    def forward(self, z,labels):
        z = z.view(z.size(0), 2)
        c = self.label_emb(labels)
        z = torch.cat([z, c], 1)
        return self.model(z)


class CVAE_fmnist(L.LightningModule):
    def __init__(
        self,
        z_dim: int = 2,
        lr: float = 0.0001,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.latent_dim = z_dim
        self.lr = lr

        # networks
        self.encoder = Encoder(in_features=784, z_dim=self.latent_dim)
        self.decoder = Decoder(in_features=self.latent_dim,out_features=784)

        # loss storage
        self.training_step_outputs = []
        self.losses = []

        # latent parameters 
        self.training_step_mu = []
        self.mu = []
        self.training_step_logvar = []
        self.logvar = []


    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        mu, log_var = self.encoder(x,labels)
        z = self.reparametrize(mu, log_var)
        recon_x = self.decoder(z,labels)
        return recon_x, mu, log_var
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = torch.add(torch.mul(eps, std), mu)
        return z
    
    def loss_function(self, recon_x, x, mu, log_var):
        recon_error = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction='sum') / x.size(0)
        kl_div = torch.sum(0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var)) / x.size(0)
        out = recon_error + kl_div
        return out
    
    def training_step(self, batch):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        recon_x, mu, log_var = self(imgs,labels)
        loss = self.loss_function(recon_x,imgs.view(imgs.size(0), -1),mu,log_var)

        values = {"train_loss": loss}
        self.log_dict(values, prog_bar = True)
        self.log('train_loss', loss, on_step=True, on_epoch=False,prog_bar=True)
        self.training_step_outputs.append(loss)
        self.training_step_mu.append(mu)
        self.training_step_logvar.append(log_var)

        return loss

    def on_train_epoch_end(self):
        loss_epoch = torch.stack(self.training_step_outputs).mean()
        self.losses.append(loss_epoch.item())
        self.training_step_outputs.clear()  # free memory

        mu_epoch = torch.cat(self.training_step_mu, dim=0).mean(dim=0)
        self.mu.append(mu_epoch.tolist())
        self.training_step_mu.clear()  # free memory

        logvar_epoch = torch.cat(self.training_step_logvar, dim=0).mean(dim=0)
        self.logvar.append(logvar_epoch.tolist())
        self.training_step_logvar.clear()  # free memory
        
        # Save generator weights at specific epochs
        save_epochs = [1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        if self.current_epoch + 1 in save_epochs:
           torch.save(self.state_dict(), f'./cvae/cvae_epoch_{self.current_epoch + 1}.pth')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
