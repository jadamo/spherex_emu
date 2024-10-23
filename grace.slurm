#!/bin/bash

#SBATCH --job-name=eft_ts
#SBATCH --output=ograce10_22.out
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:15:00


#cd spherex_emu

module load anaconda/2024.06

source ~/.bashrc

#conda activate spherexemu

python scripts/make_training_set_eft.py

