# Welcome to the COp-Net ! 

## Description

**COp-Net**, for Cell Contour Closing Operator, is a novel deep learning-based approach developed for cell instance segmentation in biological microscopy imaging. This method aims to automatically detect and fill gaps in an initial cell contour segmentation with missing information, reducing the need for manual correction for biological experts. The primary focus of the research is to apply convolutional neural network (CNN) architectures for instance segmentation in 3D electron microscopy images.


This repository contains the code and resources related to the COp-Net model, including: 
1. the source code to solve a diffusion partial differntial equation (PDE) to generate cell contour probability maps with missing informations from ground truth cell contour segmentation
2. the closing network weights
3. a python script to perform the iterative inference of our trained COp-Net
4. images to visualise the results from the experiments conducted.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)

## Installation

To run COp-Net, you'll need Python XXX and several dependencies. You can install them by following these steps:

1. Install the nnU-Netv2. (see instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md))

2. Clone the repository:
```git clone https://github.com/Florian-40/CellSegm.git
cd CellSegm 
```
3. Install the dependencies: 
```
pip install -r requirements.txt
```

## Usage 
Two Jupyter notebooks are proposed. 
- PDESolving.ipynb: generate cell contour probability maps from a private ground truth cell contour segmentation., by solving a dedicated PDE (Eq. 1 in the paper). Two time scheme are proposed: Crank-Nicolson and Forward Euler. You can apply this on your own ground truth segmentation and modify the hyper-parameters. 



## Model

**COp-Net** leverages a state-of-the-art convolutional neural network architecture (nnU-Net) for instance segmentation in an iterative inference scheme. 

For detailed information about the model architecture and training process, refer to the paper associated with this project:

**Paper Title**: Enhancing Cell Instance Segmentation in Scanning Electron Microscopy Images via a Deep Contour Closing Operator 
**Authors**: Florian Robert, Alexia Calovoulos, Laurent Facq, Fanny Decoeur, Etienne Gontier, Christophe F. Grosset, Baudouin Denis de Senneville
**Journal**: Computers in Biology and Medicine  
**Year**: 2024



## Citation

If you use COp-Net in your work, please cite the following paper:








