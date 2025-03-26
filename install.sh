#!/bin/bash

set -e  # Exit on error

########### conda stuff

# Always execute this script with bash, so that conda shell.hook works.
# Relevant conda bug: https://github.com/conda/conda/issues/7980
if [[ -z "$BASH_VERSION" ]];
then
    exec bash "$0" "$@"
fi

eval "$(conda shell.bash hook)"

echo "Creating new anaconda enviornment..."
conda create -n spherex_emu python=3.11
conda activate spherex_emu

# install the correct version of pytorch based on the device specifications 
# (ex: is there a GPU? Is the device a M1/2/3 Macbook?)

if command -v nvidia-smi &> /dev/null; then # Check for NVIDIA GPU
    echo "NVIDIA GPU found!. Installing PyTorch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
elif [[ "$(uname)" == "Darwin" ]]; then # Check if running on macOS
    echo "Device is a Mac. Installing Mac compatable version of PyTorch..."
    pip install torch
else # Otherwise, fetch only cpu-specific libraries
    echo "No GPU found. Installing CPU version of PyTorch..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# explicitely install some same package versions as the SPHEREx inference module to ensure code remains compatible
conda install numpy==1.25.2
conda install scipy=1.11.2
# install spherex_emu itself
pip install .

conda deactivate

echo "Succesfully installed spherex_emu!"