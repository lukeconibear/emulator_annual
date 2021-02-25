#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 4
#$ -l h_vmem=128G

conda activate pangeo_latest
python health_impacts_per_emission_configuration.py
