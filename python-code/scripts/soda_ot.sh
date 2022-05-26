#!/bin/bash
#SBATCH --output=soda__out.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=48Gb
#SBATCH --mail-user=frederic.beaupre.3@ulaval.ca
#SBATCH --partition=batch_48h

python python-code/otc_soda.py
