# Tabular Data Augmentation

In this repository, we share several techniques that have been used to augment the minority class of a medical dataset.
This dataset is part of a research project of Sant'Andrea Hospital about predicting Gastric Neoplastic Lesions in patients with Gastric Atrophy.


| Group | Number of Patients | Description                                  |
|-------|--------------------|----------------------------------------------|
| 1     | 297                | No lesions at any time                       |
| 2     | 22                 | No lesions at baseline, lesions at follow-up |


## Original data
For categorical variables, we show the frequency of each class. For numerical variables, we scatter plot their values.


![Original](https://github.com/msilver22/data_augmentation/blob/56939602ad8cb0b3d98a671c493d1129830ac581/tabular_data_aug/images/original_data.png)

### Classification on real data

#### Dataset partition
| Group | Training set | Test set |
|-------|--------------|----------|
| 1     | 208          | 89       |
| 2     | 15           | 7        |
#### Evaluations
| Classifier | Accuracy | F1 Score |
|------------|----------|----------|
| MLP        | 0.78     | 0.23     |

## Minority-class Augmentation
We train the models on the whole minority class. Here we compare generated data (220 samples) and real data (22 samples).

### SMOTE
![SMOTE](https://github.com/msilver22/data_augmentation/blob/835e4665a2565a8c23d9ff478531074646adc40c/tabular_data_aug/images/smote_minority.png)
### CT-GAN
![CTGAN](https://github.com/msilver22/data_augmentation/blob/490999b4720a7ee8e7adb642da641d11df31592f/tabular_data_aug/images/ctgan_minority.png)
### T-VAE
![TVAE](https://github.com/msilver22/data_augmentation/blob/65bac066725f34528d0f5f53f1b84254effb2cf6/tabular_data_aug/images/tvae_minority.png)
### GReaT
![GREAT](https://github.com/msilver22/data_augmentation/blob/d62c7d92c80e560e740c5ae4b3e0cfea1cb2337b/tabular_data_aug/images/great_minority.png)

### Classification on augmented data
We train the models only on training samples of the minority-class, in order to obtain a more balanced dataset (161 synthetic samples).
#### Dataset partition
| Group | Training set | Test set |
|-------|--------------|----------|
| 1     | 208          | 89       |
| 2     | 176          | 7        |
#### Evaluations
| Augmentation | Accuracy | F1 Score |
|--------------|----------|----------|
| None         | 0.78     | 0.23     |
| SMOTE        | 0.74     | 0.14     |
| CT-GAN       | 0.71     | 0.23     |
| T-VAE        | **0.86** | 0.29     |
| GReaT        | 0.81     | **0.3**  |






