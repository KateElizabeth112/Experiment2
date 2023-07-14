#!/bin/bash

# Launch virtual environment
source venv/bin/activate

# Install pytorch
pip3 install torch torchvision

# Install nnUNet V2
#pip install nnunetv2

# Install nnUNet V1
pip3 install --upgrade setuptools
pip3 install https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/nnunet