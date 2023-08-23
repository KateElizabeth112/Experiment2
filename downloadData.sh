#!/bin/bash
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -N download_data

curl https://zenodo.org/record/6802614/files/Totalsegmentator_dataset.zip?download=1