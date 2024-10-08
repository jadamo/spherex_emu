#/bin/bash

#SBATCH --job-name=eft_ts
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --time=08:00:00

python scripts/make_training_set_eft.py
