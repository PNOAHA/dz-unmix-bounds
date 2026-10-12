#!/bin/bash
#SBATCH --job-name=dz-p3d-merge
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --nice=30000
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
cd /public/home/panlujia/dz-unmix/paper2
export DZ_OUTDIR=results_phase3d
/public/software/Anaconda3/bin/python merge_phase3d.py

