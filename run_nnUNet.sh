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
TASK='Task302'

export nnUNet_raw_data_base=$ROOT_DIR"nnUNet_raw_data_base"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export RESULTS_FOLDER=$ROOT_DIR"RESULTS_FOLDER"

echo $nnUNet_raw_data_base
echo $nnUNet_preprocessed
echo $RESULTS_FOLDER

# Run script to generate dataset json
python3 generateDatasetJson.py -r $ROOT_DIR -n $DS -tc Task301

nnUNet_plan_and_preprocess -t 302 --verify_dataset_integrity

# Train
nnUNet_train 2d nnUNetTrainerV2 Task302 0 --npz

# Inference
#INPUT_FOLDER=$ROOT_DIR"nnUNet_raw_data_base/nnUNet_raw_data/Task301/imagesTs"
#OUTPUT_FOLDER=$ROOT_DIR"inference/Task301/fold0"

#echo $INPUT_FOLDER
#echo $OUTPUT_FOLDER

#nnUNet_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -t $TASK -m 2d -f 0
