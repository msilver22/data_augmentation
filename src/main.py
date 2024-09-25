# src/main.py

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import lightning as L
from data_loader import MNISTDataModule, FashionMNISTDataModule  
from utils.visualization import display_images

from models.cgan import CGAN  
from models.gan import Generator  

# Define a function to choose the data module dynamically
def choose_data_module(dataset_name):
    if dataset_name == 'mnist':
        return MNISTDataModule()
    elif dataset_name == 'fashion_mnist':
        return FashionMNISTDataModule()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Define a function to choose the model dynamically
def choose_model(model_name):
    if model_name == 'cgan' :
        return CGAN()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main(model_name='cgan', dataset_name='mnist', latent_dim=100, epochs=70):
    # Load the chosen data module and model
    dm = choose_data_module(dataset_name)
    model = choose_model(model_name)

    # Set up the trainer
    trainer = L.Trainer(
        max_epochs=epochs,
    )

    # Train the model
    trainer.fit(model, dm)

    # Save generator weights
    torch.save(model.generator.state_dict(), 'generator_weights_final.pth')
    print("[LOG] Generator weights saved.")

    # Plot the loss curves
    d_losses = model.d_losses
    g_losses = model.g_losses
    plot_epochs = range(0, len(g_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, d_losses, "-o", label="Discriminator loss", color="blue")
    plt.plot(plot_epochs, g_losses, "-o", label="Generator loss", color="orange")
    plt.title("Loss functions over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Load the trained generator
    g = Generator(latent_dim=latent_dim)
    g.load_state_dict(torch.load('generator_weights_final.pth'))
    g.eval()

    # Generate images
    z = torch.randn(100, latent_dim)  # 100 latent vectors
    labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)]))  # Example labels
    fake_images = g(z, labels)
    fake_images = fake_images.view(100, 1, 28, 28).detach()

    # Display generated images
    display_images(fake_images, n_cols=10)

# Entry point for the script
if __name__ == '__main__':
    # Here you can pass parameters dynamically, for example:
    main(model_name='cgan', dataset_name='mnist', latent_dim=100, epochs=70)
