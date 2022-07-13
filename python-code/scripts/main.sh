#!/bin/bash
#SBATCH --output=main_out.log
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --mem=48Gb
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --partition=gpu_48h

python main.py
