#!/bin/bash
#SBATCH --output=baselines.log
#SBATCH --time=96:00:00
#SBATCH --gres=gpu
#SBATCH --mem=48Gb
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --partition=gpu_96h

python main.py
