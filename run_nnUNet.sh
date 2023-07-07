#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4 # Number of CPU Cores
#SBATCH -p gpushigh # Partition (queue)
#SBATCH --gres gpu:1 # gpu:n, where n = number of GPUs
#SBATCH --mem 20G # memory pool for all cores
#SBATCH --nodelist monal03 # SLURM node
#SBATCH --output=slurm.%N.%j.log # Standard output and error log
# Launch virtual environment
source venv/bin/activate

# Set environment variables
#ROOT_DIR='/Users/katecevora/Documents/PhD/data/TotalSegmentator_nnUNet/'
ROOT_DIR='/vol/biomedic3/kc2322/data/TotalSegmentator_nnUNetv1/'
DS='Task301'

export nnUNet_raw_data_base=$ROOT_DIR"nnUNet_raw_data_base"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export RESULTS_FOLDER=$ROOT_DIR"RESULTS_FOLDER"

echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results

# Run script to generate dataset json
#python3 generateDatasetJson.py -r $ROOT_DIR -n $DS -tc Task301

#nnUNet_plan_and_preprocess -t 301 --verify_dataset_integrity

# Train
nnUNet_train 3d_fullres nnUNetTrainerV2 Task301 0 --npz

# Inference
#INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/Dataset200_AMOS/imagesVaSorted"
#OUTPUT_FOLDER=$ROOT_ROOT"inference/preds"

#echo $INPUT_FOLDER
#echo $OUTPUT_FOLDER

#nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d 200 -c 2d -f 0 --verbose