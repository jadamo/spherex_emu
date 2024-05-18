#!/bin/bash

#SBATCH --job-name=arraytest
#SBATCH --output=array.out
#SBATCH --account=timeifler
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5gb
#SBATCH --time=00:30:00
#SBATCH --array=1-3

#module load python/3.11 anaconda

#module load openmpi3/3.1.4

source /home/u14/gibbins/.bashrc

#conda activate spherexemu

python linps.py
