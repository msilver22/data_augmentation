# data_augmentation

data_augmentation/
│
├── notebooks/               
│   ├── basic_image_augmentation_logo_sapienza.ipynb    # Basic image augmentation su logo Sapienza
│   ├── gan_mnist_training.ipynb                        # GAN per MNIST
│
├── src/                      # Codice sorgente riutilizzabile
│   ├── data_loader.py         # Funzioni per caricare dataset
│   ├── models/                # Modelli di machine learning
│   │   ├── gan.py             # Definizione GAN
│   │   ├── wgan.py            # Definizione WGAN
│   │   ├── cgan.py            # Definizione CGAN
│   │   ├── vae.py             # Definizione VAE
│   │   └── cvae.py            # Definizione CVAE
│   └── utils/                 # Funzioni di utilità (per augmentazione o altro)
│       ├── augmentation.py    # Funzioni per image augmentation
│       └── visualization.py   # Funzioni per visualizzazione risultati
│
├── data/                     # Dati o modelli pre-addestrati (se non troppo pesanti)
│   ├── logo_sapienza.png      # Logo di Sapienza usato per image augmentation
│   └── mnist/                # Eventuali dataset pre-processati
│
├── results/                  # Salvataggio di immagini o modelli generati
│   ├── gan_mnist/            # Risultati di addestramento GAN su MNIST
│   ├── wgan_fashionmnist/    # Risultati di WGAN su FashionMNIST
│   └── vae_mnist/            # Risultati di VAE su MNIST e FashionMNIST
│
├── .gitignore                # File per ignorare dati o output non necessari
├── README.md                 # Documentazione principale del progetto
