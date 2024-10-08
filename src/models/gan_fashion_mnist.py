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
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class GAN_fmnist(L.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 0.0002,
        k : int=5,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.k = k

        # networks
        self.generator = Generator(in_features=self.latent_dim, out_features=784)
        self.discriminator = Discriminator(in_features=784,out_features=1)

        # loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.training_step_outputs_d = []
        self.training_step_outputs_g = []
        self.d_losses = []
        self.g_losses = []

    def forward(self, z):
        return self.generator(z)

    def real_loss(self, predicted_outputs, loss_fn, device):

        # Targets are set to 1 here because we expect prediction to be
        # 1 (or near 1) since samples are drawn from real dataset
        batch_size = predicted_outputs.size(0)
        targets = torch.ones(batch_size).to(device)
        real_loss = loss_fn(predicted_outputs.squeeze(), targets)

        return real_loss

    def fake_loss(self, predicted_outputs, loss_fn, device):

        # Targets are set to 0 here because we expect prediction to be
        # 0 (or near 0) since samples are generated fake samples
        batch_size = predicted_outputs.size(0)
        targets = torch.zeros(batch_size).to(device)
        fake_loss = loss_fn(predicted_outputs.squeeze(), targets)

        return fake_loss

    def training_step(self, batch):
        imgs, _ = batch
        imgs = imgs.to(device)

        optimizer_g, optimizer_d = self.optimizers()

        ## ----------------------------------------------------------------
        ## Train discriminator using real and then fake MNIST images,
        ## then compute the total-loss and back-propogate the total-loss
        ## ----------------------------------------------------------------

        self.toggle_optimizer(optimizer_d)

        for _ in range(self.k):
          # Real MNIST images
          d_real_logits_out = self.discriminator(imgs)
          d_real_loss = self.real_loss(d_real_logits_out, self.loss_fn, device)
          # Fake images
          with torch.no_grad():
            # Generate a batch of random latent vectors
            z = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim))
            z = torch.from_numpy(z).float().to(device)
            # Generate batch of fake images
            fake_images = self(z)
          # feed fake-images to discriminator and compute the
          # fake_loss (i.e. target label = 0)
          d_fake_logits_out = self.discriminator(fake_images)
          d_fake_loss = self.fake_loss(d_fake_logits_out, self.loss_fn, device)

          # Compute total discriminator loss
          d_loss = d_real_loss + d_fake_loss
          self.log('d_loss', d_loss, on_step=True, prog_bar=True, logger=True)
          self.manual_backward(d_loss)
          optimizer_d.step()
          optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)

        ## ----------------------------------------------------------------
        ## Train generator, compute the generator loss which is a measure
        ## of how successful the generator is in tricking the discriminator
        ## and finally back-propogate generator loss
        ## ----------------------------------------------------------------

        self.toggle_optimizer(optimizer_g)

        # Generate a batch of random latent vectors
        z = np.random.uniform(-1, 1, size=(self.batch_size, self.hparams.latent_dim))
        z = torch.from_numpy(z).float().to(device)
        # Generate a batch of fake images, feed them to discriminator
        # and compute the generator loss as real_loss
        # (i.e. target label = 1)
        fake_images = self.generator(z)
        g_logits_out = self.discriminator(fake_images)
        g_loss = self.real_loss(g_logits_out, self.loss_fn, device)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.training_step_outputs_d.append(d_loss)
        self.training_step_outputs_g.append(g_loss)

    def on_train_epoch_end(self):
        d_loss_epoch = torch.stack(self.training_step_outputs_d).mean()
        g_loss_epoch = torch.stack(self.training_step_outputs_g).mean()
        self.d_losses.append(d_loss_epoch.item())
        self.g_losses.append(g_loss_epoch.item())
        self.training_step_outputs_d.clear()  # free memory
        self.training_step_outputs_g.clear()

        # Save generator weights at specific epochs
        save_epochs = [1, 5, 10, 20, 40, 60, 80, 100, 200]
        if self.current_epoch + 1 in save_epochs:
           torch.save(self.generator.state_dict(), f'generator_epoch_{self.current_epoch + 1}.pth')

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
