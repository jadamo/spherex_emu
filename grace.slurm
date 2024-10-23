#!/bin/bash

#SBATCH --job-name=eft_ts
#SBATCH --output=ograce100k.out
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=94
#SBATCH --time=10:00:00


#cd spherex_emu

module load anaconda/2024.06

source ~/.bashrc

#conda activate spherexemu

python scripts/make_training_set_eft.py

