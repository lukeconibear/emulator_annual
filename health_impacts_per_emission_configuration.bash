#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=02:00:00
#$ -pe smp 1
#$ -l h_vmem=128G


conda activate pangeo_latest
python health_impacts_per_emission_configuration.py
