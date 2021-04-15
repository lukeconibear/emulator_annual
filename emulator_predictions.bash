#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=4G


conda activate pangeo_latest
python /nobackup/earlacoa/machinelearning/scripts_annual/emulator_predictions.py /nobackup/earlacoa/machinelearning/data_annual/emulators/PM2_5_DRY /nobackup/earlacoa/machinelearning/data_annual/predictions/PM2_5_DRY/
