#!/bin/bash
#SBATCH --output=soda__out.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=100Gb
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --partition=gpu_48h

python main_rings.py
