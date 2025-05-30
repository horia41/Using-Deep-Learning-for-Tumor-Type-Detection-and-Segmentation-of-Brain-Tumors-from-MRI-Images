# Bachelor Thesis Project

# Using Deep Learning For Tumor Type Detection And Segmentation Of Gliomas, Meningiomas, And Pituitary Adenomas From Magnetic Resonance Imaging Scans 
# *A Model Architecture Analysis*

## Overview

This project implements and evaluates multiple deep learning architectures variants for simultaneous brain tumor segmentation and classification using T1-weighted contrast-enhanced MRI scans. It investigates how integrating classification and segmentation tasks mutually affect each other's performance, explores different fusion strategies, and includes statistical and explainability analyses.

## Features

- Segmentation-only, classification-only, and multi-task learning models based on FCN-ResNet50 backbone
- Fusion techniques including “project and add” classification embeddings into segmentation
- Comprehensive experiments comparing task synergy and trade-offs
- Occlusion sensitivity analysis for interpretability
- Scripts for training, validation, and evaluation with PyTorch Lightning

## Dataset

- Figshare Brain Tumor Dataset: 3064 MRI slices with expert masks and three tumor classes (meningioma, glioma, pituitary adenoma)
- Preprocessing includes resizing to 256×256, normalization, and binary mask thresholding

## Model Variants

1. Segmentation-only (Model A)
2. Classification-only (Model B)
3. Frozen-features classification (Model C)
4. Naïve joint training (Model D)
5. Project-and-add fusion of class logits (Model E)
6. Segmentation embeddings for classification (Model F)
7. Penultimate classification embedding fusion (Model G)

## Project Structure
- `checkpoints` — Model checkpoints folder
- `data_figshare` — Dataset folder
- `images` — Plots folder from code
- `lightning_logs` — PyTorch Lightning folder, ignore this
- `Explainable_AI_Analysis.ipynb` — Jupyter Notebook for Occlusion Analysis
- `Model_A_Segmentation_Only.ipynb` — Jupyter Notebook for Model A
- `Model_B_Classification_Only.ipynb` — Jupyter Notebook for Model B
- `Model_C_Staged_Training.ipynb` — Jupyter Notebook for Model C
- `Models_D_E_F_G.ipynb` — Jupyter Notebook for Models D, E, F, G
- `README.md` — README file
- `Statistical_Analysis.ipynb` — Jupyter Notebook for conducted statistical analysis


## Results Summary

| Model                              | Seg. Dice | Classification Accuracy |
|------------------------------------|-----------|-------------------------|
| Segmentation-only (A)              | 0.7311    | –                       |
| Classification-only (B)            | –         | 0.9052                  |
| Frozen-features classification (C) | 0.7309    | 0.7157                  |
| Naïve joint training (D)           | 0.7344    | 0.9232                  |
| Project-and-add fusion (E)         | <b>0.7482 | 0.9461                  |
| Segmentation-informed cls (F)      | 0.7517    | <b>0.9477               |
| Penultimate embeddings fusion (G)  | 0.7408    | 0.9592                  |

## How to Run

1. Access the following **Google Drive** link to download the Dataset files and Model checkpoints:
   [Dataset and Models](https://drive.google.com/drive/folders/1v7aCJuNBhcFqmI3yGVbRx_EidV7j-9T6?usp=sharing).
    Structure of **Google Drive** folder :
    - `Models` — Model checkpoints folder
    - `dataset_zipped.zip` — Dataset folder
2. In the `Models` folder from **Google Drive**, download and then take all model checkpoints and put them in the `checkpoints` folder from this project.
3. Unzip the `dataset_zipped.zip` from **Google Drive** and put everything from it in the `data_figshare` folder from this project.
4. At this point, you can run any notebook from the project.

## Author

**[Horia Ionescu](mailto:h.ionescu@student.maastrichtuniversity.nl)**  
*Supervisor:* [Enrique Hortal Quesada](mailto:enrique.hortal@maastrichtuniversity.nl)  
[Department of Advanced Computing Sciences](https://www.maastrichtuniversity.nl/research/department-advanced-computing-sciences)  
Faculty of Science and Engineering  
Maastricht University, The Netherlands
