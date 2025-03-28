# spherex_emu

This repository contains code to create a neural network emulator that outputs redshift-space galaxy power spectrum multipoles given a set of input cosmology + galaxy bias parameters. Said emulator is able to generate multipoles for multiple tracer and redhshift bins simultaniously, and is primarily meant for use in SPHEREx likelihood inference studies. 

## Installing the code

This package works on both Linux and MacOS (intel and arm64) platforms, and has so-far been tested using **Python 3.11**. There are two methods to install the code.

### Preliminaries

1. To enable GPU functionality for network training, make sure you have CUDA installed (or python 3.8+ if using apple silicon).
2. You will need some way to generate galaxy power spectrum multipoles to generate training sets. One option is to download and install both [ps_1loop](https://github.com/archaeo-pteryx/ps_1loop) and [ps_theory_calculator](https://github.com/archaeo-pteryx/ps_theory_calculator). You might need to request access to those repositories, in which case you can contact Yosuke Kobayashi (yosukekobayashi@arizona.edu). We have also included a version of [FAST-PT](https://github.com/jablazek/FAST-PT) to satisfy this requirnment.
3. Download this repository to your location of choice.

### Automated Install (recommended)

In the base directory, simply run `install.sh` in the terminal. This script will create a new anaconda enviornment, fetch the corresponding version of PyTorch, and install the code, all automatically.

### Manual Install 

1. install the corresponding [PyTorch version](https://pytorch.org/get-started/locally/). If your machine doesn't have a GPU, you can skip this step.
2. In the base directory, run `python -m pip install .`, which should install this repository as a package and all required dependencies.

To run the provided unit-tests, you can run the following command in the base repo directory,

`python -m pytest tests`

## Running the code

TODO: Update this section

## Authors

- Joe Adamo (primary) (jadamo@arizona.edu)
- Annie Moore
- Grace Gibbins
