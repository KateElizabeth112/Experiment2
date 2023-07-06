#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 20G # memory pool for all cores
#SBATCH --nodelist monal04 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log
# Launch virtual environment
source venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/'
ROOT_DIR='/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNet/'
DS='Dataset301_Set1'

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Run script to generate dataset json
python3 generateDatasetJson.py -r $ROOT_DIR -n $DS

nnUNetv2_plan_and_preprocess -d 301 -c 3d_fullres

nnUNetv2_extract_fingerprint -d 301
nnUNetv2_plan_experiment -d 301 -c 3d_fullres -np 3
nnUNetv2_preprocess -d 301 -c 3d_fullres -np 3

# Plan and preprocess data
#nnUNetv2_plan_and_preprocess -d 301 -c 3d_fullres

# Train
#nnUNetv2_train 301 3d_fullres 0

# Inference
#INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/Dataset200_AMOS/imagesVaSorted"
#OUTPUT_FOLDER=$ROOT_ROOT"inference/preds"

#echo $INPUT_FOLDER
#echo $OUTPUT_FOLDER

#nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d 200 -c 2d -f 0 --verbose