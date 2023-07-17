#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 32G # memory pool for all cores
#SBATCH --nodelist monal03 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log
# Launch virtual environment
source venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/'
ROOT_DIR='/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNetv1/'
TASK='Task601_Mixed'

export nnUNet_raw_data_base=$ROOT_DIR"nnUNet_raw_data_base"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export RESULTS_FOLDER=$ROOT_DIR"RESULTS_FOLDER"

echo $nnUNet_raw_data_base
echo $nnUNet_preprocessed
echo $RESULTS_FOLDER

# Run script to generate dataset json
echo "Generating dataset..."
#python3 generateDatasetJson.py -r $ROOT_DIR -n $DS -tc Task301

echo "Planning and precprocessing..."
#nnUNet_plan_and_preprocess -t 601 --verify_dataset_integrity

nnUNet_train 2d nnUNetTrainerV2 $TASK 0

# Train
"""
echo 'Training...'
echo 'Fold 1'
nnUNet_train 2d nnUNetTrainerV2 $TASK 1

echo 'Fold 2'
nnUNet_train 2d nnUNetTrainerV2 $TASK 2

echo 'Fold 3'
nnUNet_train 2d nnUNetTrainerV2 $TASK 3

echo 'Fold 4'
nnUNet_train 2d nnUNetTrainerV2 $TASK 4
"""