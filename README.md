# Data Augmentation 

This project explores various data augmentation techniques and image generation models, such as **GAN**, **WGAN**, **CGAN**, **VAE**, and **CVAE** using datasets like **MNIST** and **FashionMNIST**. 
It includes Jupyter notebooks with step-by-step implementations and reusable source code to facilitate exploration and understanding of augmentation techniques.

## Repository Structure

      data_augmentation/ 
      │
      ├── notebooks/                
      │   ├── Logo_sapienza_augmentation.ipynb    
      │   ├── gan_mnist_training.ipynb                        
      │
      ├── src/                      
      │   ├── data_loader.py        
      │   ├── models/                
      │   │   ├── gan.py             
      │   │   ├── wgan.py           
      │   │   ├── cgan.py            
      │   │   ├── vae.py            
      │   │   └── cvae.py           
      │   └── utils/                 
      │       ├── augmentation.py   
      │       └── visualization.py  
      │
      ├── data/                    
      │   ├── logo_sapienza.png     
      │   │── MNIST/                
      │   └── FashionMNIST  
      │
      ├── photos/
      │   ├── Sapienza_logos/    
      │   ├── gan_mnist/            
      │   ├── wgan_fashionmnist/   
      │   └── vae_mnist/           
      │
      ├── .gitignore                # File to ignore unnecessary data or output
      └──README.md                 # Main project documentation

