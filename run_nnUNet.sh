#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 12 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --nodelist monal03 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log
# Launch virtual environment
source /vol/biomedic3/kc2322/code/AMOS_3D/venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/'
ROOT_DIR='/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet/'
TASK='Dataset301_Set1'

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Plan and preprocess data
nnUNetv2_plan_and_preprocess -d 301 -c 3d_fullres -np 3 --verify_dataset_integrity

# Train
nnUNetv2_train 301 3d_fullres all