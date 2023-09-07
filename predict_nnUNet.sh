#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=12:mem=120gb:ngpus=1:gpu_type=RTX6000
#PBS -N nnUNet_AMOS_predict

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

# Set environment variables
ROOT_DIR='/rds/general/user/kc2322/home/data/TotalSegmentator/'
DATASET="Dataset302_Set2"

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

# Inference
INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/"$DATASET"/imagesTs"
OUTPUT_FOLDER=$ROOT_DIR"inference/"$DATASET"/all"

nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d 302 -c 3d_fullres -f all

# Run python script to evaluate results
python3 processResults.py -d $DATASET