#!/bin/bash

# Launch virtual environment
source venv/bin/activate

# Install pytorch
pip3 install torch torchvision

# Install nnUNet
pip install nnunetv2

# install other python packages
pip3 install nibabel
pip3 install numpy
pip3 install matplotlib