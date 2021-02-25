#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=4G

conda activate pangeo_latest
python /nobackup/earlacoa/machinelearning/scripts_annual/emulator_predictions.py /nobackup/earlacoa/machinelearning/data_annual/emulators/o3_6mDM8h /nobackup/earlacoa/machinelearning/data_annual/predictions/o3_6mDM8h/
