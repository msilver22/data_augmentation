# Data Augmentation 

This project explores various data augmentation techniques and image generation models, such as **GAN**, **WGAN**, **CGAN**, **VAE**, and **CVAE** using datasets like **MNIST** and **FashionMNIST**. 
It includes Jupyter notebooks with step-by-step implementations and reusable source code to facilitate exploration and understanding of augmentation techniques.

## Repository Structure

'''data_augmentation/ 
│
├── notebooks/                # Contains Jupyter notebooks
│   ├── basic_image_augmentation_logo_sapienza.ipynb    # Basic image augmentation on Sapienza logo
│   ├── gan_mnist_training.ipynb                        # GAN for MNIST
│   ├── wgan_fashionmnist.ipynb                         # WGAN for FashionMNIST
│   ├── gan_fashionmnist.ipynb                          # GAN for FashionMNIST
│   ├── cgan_mnist_fashionmnist.ipynb                   # CGAN for MNIST and FashionMNIST
│   ├── vae_mnist_fashionmnist.ipynb                    # VAE for MNIST and FashionMNIST
│   └── cvae_mnist_fashionmnist.ipynb                   # CVAE for MNIST and FashionMNIST
│
├── src/                      # Reusable source code
│   ├── data_loader.py         # Functions to load datasets
│   ├── models/                # Machine learning models
│   │   ├── gan.py             # GAN definition
│   │   ├── wgan.py            # WGAN definition
│   │   ├── cgan.py            # CGAN definition
│   │   ├── vae.py             # VAE definition
│   │   └── cvae.py            # CVAE definition
│   └── utils/                 # Utility functions (for augmentation or other purposes)
│       ├── augmentation.py    # Functions for image augmentation
│       └── visualization.py   # Functions for visualizing results
│
├── data/                     # Data or pre-trained models (if not too heavy)
│   ├── logo_sapienza.png      # Sapienza logo used for image augmentation
│   └── mnist/                # Any pre-processed datasets
│
├── results/                  # Saved images or generated models
│   ├── gan_mnist/            # GAN training results on MNIST
│   ├── wgan_fashionmnist/    # WGAN results on FashionMNIST
│   └── vae_mnist/            # VAE results on MNIST and FashionMNIST
│
├── .gitignore                # File to ignore unnecessary data or output
├── LICENSE                   # Project license
├── README.md                 # Main project documentation
└── CONTRIBUTING.md           # Contributor guidelines (optional)'''

