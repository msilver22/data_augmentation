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
    if model_name == 'vae' and dataset_name=='mnist':
        from models.vae_mnist import VAE_mnist
        return VAE_mnist()
    elif model_name == 'vae' and dataset_name=='fmnist':
        from models.vae_fashion_mnist import VAE_fmnist
        return VAE_fmnist()
    elif model_name == 'cvae' and dataset_name=='mnist':
        from models.cvae_mnist import CVAE_mnist
        return CVAE_mnist()
    elif model_name == 'cvae' and dataset_name=='fmnist':
        from models.cvae_fashion_mnist import CVAE_fmnist
        return CVAE_fmnist()
    else:
        raise ValueError(f"Unknown model or dataset: {model_name}, {dataset_name}")


def main(model_name='cgan', dataset_name='mnist', z_dim=2, epochs=100):
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
    torch.save(model.generator.state_dict(), 'vae_weights_final.pth')
    print("[LOG] VAE weights saved.")

    # Plot the loss function
    losses = model.losses
    plot_epochs = range(0, len(losses))

    plt.figure(figsize=(10, 6))
    plt.plot(plot_epochs, losses, "-o", label="Loss", color="blue")
    plt.title("Loss functions over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("VAE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Load the trained generator
    v = choose_model(model_name, dataset_name)
    v.load_state_dict(torch.load('vae_weights_final.pth'))
    v.eval()

    # Generate images
    if model_name == 'vae':
       z = torch.randn(100,2)
       fake_images = v.decoder(z)
       fake_images = fake_images.view(100, 1, 28, 28).detach()
    elif model_name =='cvae':
        z = torch.randn(100,2)
        labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)]))
        fake_images = v.decoder(z,labels)
        fake_images = fake_images.view(100, 1, 28, 28).detach()

    #Visualize images    
    display_images(fake_images, n_cols=10)

if __name__ == '__main__':
    main(model_name='cgan', dataset_name='mnist', z_dim=2, epochs=100)
