#!/bin/bash

#SBATCH --job-name=eft_ts
#SBATCH --output=ograce10_1210_1210_1210_1210_1210_1210_1210_1210_1210_1210_12.out
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --time=14:00:00


#cd spherex_emu

module load anaconda/2024.06

source ~/.bashrc

#conda activate spherexemu

python scripts/make_training_set_eft.py
