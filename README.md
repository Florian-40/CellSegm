# Welcome to the COp-Net ! 

## Description

**COp-Net**, for Cell Contour Closing Operator, is a novel deep learning-based approach developed for cell instance segmentation in electron microscopy imaging. This method aims to automatically detect and fill gaps in an initial cell contour segmentation with missing information, reducing the need for manual correction for biological experts. 


This repository contains the code and resources related to the COp-Net model, including: 
1. the source code to solve a diffusion partial differential equation (PDE) to generate cell contour probability maps with missing informations from ground truth cell contour segmentation
2. the COp-Net weights
3. a python script to perform the iterative inference of our trained and publicly available COp-Net
4. images to visualise the results from the experiments conducted.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Reference](#Reference and citation)

## Installation

To run COp-Net, you'll need Python 3.9 and several dependencies. You can install them by following these steps:

1. Create conda environment:
```
conda create -y -n cellsegm python pip
conda activate cellsegm
```

2. Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip).

3. Install the nnU-Netv2. (see instructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for more details). 
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

4. Clone the repository:
```
git clone https://github.com/Florian-40/CellSegm.git
cd CellSegm 
```

5. Install the dependencies: 
```
pip install -r requirements.txt
```

6. Download Cop-Net weights: 
```
curl -o COp_Net/nnUNetv2.zip https://www.math.u-bordeaux.fr/~frobert002/documents/Cop_Net_weights/nnUNetv2.zip
```

4. Unzip model weights:
```
unzip -q ./COp_Net/nnUNetv2.zip -d ./COp_Net/ && rm -rf ./COp_Net/__MACOSX
```


## Usage 
Two Jupyter notebooks are proposed. 
- PDESolving.ipynb: generate cell contour probability maps from a private ground truth cell contour segmentation., by solving a dedicated PDE (Eq. 1 in the paper). Two time scheme are proposed: Crank-Nicolson and Forward Euler. You can apply this on your own ground truth segmentation and modify the hyper-parameters. 

- Test.ipynb: test iterative Cop-Net inference on a 2D slice of our testing dataset or apply it on your own 2D or 3D dataset.



## Model

**COp-Net** leverages a state-of-the-art convolutional neural network architecture (nnU-Net) for instance segmentation in an iterative inference scheme. 

For detailed information about the model architecture and training process, refer to the paper associated with this project:

**Paper Title**: Enhancing Cell Instance Segmentation in Scanning Electron Microscopy Images via a Deep Contour Closing Operator 
**Authors**: Florian Robert, Alexia Calovoulos, Laurent Facq, Fanny Decoeur, Etienne Gontier, Christophe F. Grosset, Baudouin Denis de Senneville
**Journal**: Computers in Biology and Medicine  
**Year**: 2025



## Reference and citation
For detailed information about the model architecture and training process, refer to the following publication :

Florian Robert, Alexia Calovoulos, Laurent Facq, Fanny Decoeur, Etienne Gontier, Christophe F. Grosset, Baudouin Denis de Senneville, “Enhancing cell instance segmentation in scanning electron microscopy images via a deep contour closing operator,” Computers in Biology and Medicine, vol. 190, p. 109972, 2025, ISSN 0010-4825, [https://doi.org/10.1016/j.compbiomed.2025.109972](https://www.sciencedirect.com/science/article/pii/S0010482525003233)


If you use COp-Net in your work, please cite the following publication:

```bibtex
@article{robert2025,
title = {Enhancing cell instance segmentation in scanning electron microscopy images via a deep contour closing operator},
journal = {Computers in Biology and Medicine},
volume = {190},
pages = {109972},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2025.109972},
url = {https://www.sciencedirect.com/science/article/pii/S0010482525003233},
author = {Florian Robert and Alexia Calovoulos and Laurent Facq and Fanny Decoeur and Etienne Gontier and Christophe F. Grosset and Baudouin {Denis de Senneville}}
}
```








