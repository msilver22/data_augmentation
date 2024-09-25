import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import lightning as L
from data_loader import MNISTDataModule, FashionMNISTDataModule  
from utils.visualization import display_images 

def choose_data_module(dataset_name):
    if dataset_name == 'mnist':
        return MNISTDataModule()
    elif dataset_name == 'fashion_mnist':
        return FashionMNISTDataModule()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def choose_model(model_name,dataset_name):
    if model_name == 'gan' and dataset_name=='mnist':
        from models.gan_mnist import GAN_mnist, Generator
        return GAN_mnist()
    elif model_name == 'gan' and dataset_name=='fmnist':
        from models.gan_fashion_mnist import GAN_fmnist, Generator
        return GAN_fmnist()
    elif model_name == 'wgan' and dataset_name=='fmnist':
        from models.wgan_fashion_mnist import WGAN_fmnist, Generator
        return WGAN_fmnist()
    elif model_name == 'cgan' and dataset_name=='mnist':
        from models.cgan_mnist import CGAN_mnist, Generator
        return CGAN_mnist()
    elif model_name == 'cgan' and dataset_name=='fmnist':
        from models.cgan_fashion_mnist import CGAN_fmnist, Generator
        return CGAN_mnist()
    else:
        raise ValueError(f"Unknown model or dataset: {model_name}, {dataset_name}")

def main(model_name='cgan', dataset_name='mnist', latent_dim=100, epochs=100):
    # Load the chosen data module and model
    dm = choose_data_module(dataset_name)
    model = choose_model(model_name, dataset_name)

    # Set up the trainer
    trainer = L.Trainer(
        max_epochs=epochs,
    )

    # Train the model
    trainer.fit(model, dm)

    # Save generator weights
    torch.save(model.generator.state_dict(), 'generator_weights_final.pth')
    print("[LOG] Generator weights saved.")

    # Plot the loss functions
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
    g = Generator()
    g.load_state_dict(torch.load('generator_weights_final.pth'))
    g.eval()

    # Generate images
    if model_name == 'gan' or model_name == 'wgan':
        z = np.random.uniform(-1, 1, size=(100, 100))
        z = torch.from_numpy(z).float()
        fake_images = g(z)
        fake_images = fake_images.view(100, 1, 28, 28).detach()
    elif model_name =='cgan':
        z = torch.randn(100,100)
        labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)]))
        fake_images = g(z,labels)
        fake_images = fake_images.view(100, 1, 28, 28).detach()
        
    #Visualize images    
    display_images(fake_images, n_cols=10)

if __name__ == '__main__':
    main(model_name='cgan', dataset_name='mnist', latent_dim=100, epochs=70)
