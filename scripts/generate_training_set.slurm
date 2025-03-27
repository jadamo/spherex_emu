#!/bin/bash
#BATCH --job-name=Training-Set-Gen
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jadamo@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=94
#SBATCH --ntasks-per-node=1
####SBATCH --switches=1
#SBATCH --time=75:00:00   
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --qos=part_qos_standard
##SBATCH --qos=user_qos_timeifler
#SBATCH --output=training-set.out

module load gnu8 openmpi3 python/3.11 anaconda/2022
#source /home/u12/jadamo/virtual-env/bin/activate
source /home/u12/jadamo/.bashrc
micromamba activate spherex_cpu

cd /home/u12/jadamo/SPHEREx/spherex_emu/scripts

time python3 make_training_set_eft.py
