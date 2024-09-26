# Data Augmentation 

This repository explores various data augmentation techniques and image generation models, such as **GAN**, **WGAN**, **CGAN**, **VAE**, and **CVAE** using datasets like **MNIST** and **FashionMNIST**. 
It includes Jupyter notebooks with step-by-step implementations and reusable source code to facilitate exploration and understanding of augmentation techniques.
In addition, in the folder *"tabular_data_aug"* we explore several augmentation tecniques for tabular data, such as **SMOTE**, **CT-GAN**, **T-VAE** and **GReaT**.

## Repository Structure

      data_augmentation/ 
      │
      ├── notebooks/                
      │   ├── wgan_fashion_mnist.ipynb 
      │   ├── gan_mnist.ipynb 
      │   └── logo_sapienza_augmentation.ipynb  
      │
      ├── src/                      
      │   ├── data_loader.py        
      │   ├── models/                
      │   │   ├── gan_mnist.py  
      │   │   ├── gan_fashion_mnist.py   
      │   │   ├── wgan_fashion_mnist.py           
      │   │   ├── cgan_mnist.py   
      │   │   ├── cgan_fashion_mnist.py  
      │   │   ├── vae_mnist.py 
      │   │   ├── vae_fashion_mnist.py 
      │   │   ├── cvae_mnist.py 
      │   │   └── cvae_fashion_mnist.py  
      │   ├── main_gan.py   
      │   ├── main_vae.py  
      │   └── utils/                  
      │       └── visualization.py  
      │
      ├── photos/
      │   ├── Sapienza_logos/    
      │   ├── gan/            
      │   ├── wgan/ 
      │   ├── cgan/ 
      │   └── vae/  
      │
      │
      ├── tabular_data_aug/                       
      │   │── src/ 
      │   │   ├── data_loader.py        
      │   │   ├── models/
      │   │   │   └── mlp_classifier.py/  
      │   │   ├── main_SMOTE.py 
      │   │   ├── main_CTGAN.py 
      │   │   ├── main_TVAE.py 
      │   │   └── main_GREAT.py  
      │   └── README.md  
      │
      ├── .gitignore                
      ├── README.md                 
      └── requirements.txt                 

## MNIST experiment
### GAN
![GAN](https://github.com/msilver22/data_augmentation/blob/449db8b1605d55e2e6bbd3822b8ca696557bcbea/photos/gan/mode_collapse.png)
### CGAN
![CGAN](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/cgan/cgan_mnist.png)
### VAE
![VAE](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/vae/vae_mnist.png)
### CVAE 
![CVAE](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/vae/cvae_mnist.png)

## FashionMNIST experiment
### GAN
![GAN](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/gan/gan_fmnist.png)
### WGAN
![WGAN](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/wgan/wgan.png)
### CGAN
![CGAN](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/cgan/cgan_fmnist.png)
### VAE
![VAE](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/vae/vae_fmnist.png)
### CVAE
![CVAE](https://github.com/msilver22/data_augmentation/blob/4a3547ce97160eb7e9127af8139641012bfcc971/photos/vae/cvae_fmnist.png)





