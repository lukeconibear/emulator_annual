#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=05:00:00
#$ -pe smp 1
#$ -l h_vmem=32G


conda activate pangeo_latest
python create_o3_metric.py
