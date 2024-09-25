import torch
import torch.nn as nn
import lightning as L
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count() / 2)
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class Generator(nn.Module):
    def __init__(self, in_features=100, out_features=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, out_features),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        z = z.view(z.size(0), -1)
        return z

  
class Discriminator(nn.Module):
    def __init__(self, in_features=784, out_features=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,out_features)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

  class WGAN(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 0.00005,
        batch_size: int = BATCH_SIZE,
        n_critic = 5,
        clip_value = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.n_critic = n_critic
        self.clip_value = clip_value

        # networks
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        # loss
        self.training_step_outputs_wasserstein = []
        self.w_distance = []

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch):
        imgs, _ = batch
        imgs = imgs.to(device)

        optimizer_g, optimizer_d = self.optimizers()

        ## ----------------------------------------------------------------
        ## Train discriminator to compute Wasserstein distance
        ## ----------------------------------------------------------------

        for _ in range(self.n_critic):
            # Sample random noise vectors
            z = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim))
            z = torch.from_numpy(z).float().to(device)
            # Generate fake images
            fake_images = self(z)
            # Compute Wasserstein distance
            real_logits = self.discriminator(imgs)
            fake_logits = self.discriminator(fake_images)
            d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            #Making the discriminator K-Lipschitz
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

        self.log('wasserstein_d', -d_loss, on_step=True, prog_bar=True, logger=True)
        self.training_step_outputs_wasserstein.append(-d_loss)

        ## ----------------------------------------------------------------
        ## Train generator to minimize the Wasserstein distance
        ## ----------------------------------------------------------------

        # Sample random noise vectors
        z = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim))
        z = torch.from_numpy(z).float().to(device)
        # Generate fake images
        fake_images = self(z)
        # Compute Wasserstein distance
        fake_logits = self.discriminator(fake_images)
        g_loss = -torch.mean(fake_logits)
        #self.log("loss_G", g_loss, prog_bar=True)
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    def on_train_epoch_end(self):
        w_distance_epoch = torch.stack(self.training_step_outputs_wasserstein).mean()
        self.w_distance.append(w_distance_epoch.item())
        self.training_step_outputs_wasserstein.clear()  # free memory

        # Save generator weights at specific epochs
        save_epochs = [1, 5, 10, 20, 40, 60, 80, 100, 200]
        if self.current_epoch + 1 in save_epochs:
           torch.save(self.generator.state_dict(), f'generator_epoch_{self.current_epoch + 1}.pth')

    def configure_optimizers(self):
        lr = self.lr
        opt_g = optim.RMSprop(self.generator.parameters(), lr=lr)
        opt_d = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []
