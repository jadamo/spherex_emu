# spherex_emu

This repository contains code to create a neural network emulator that outputs redshift-space galaxy power spectrum multipoles given a set of input cosmology + bias parameters. Said emulator is able to generate multipoles for multiple sample and redhshift bins simultaniously. 

## Installing the code

This package should run on both Linux and MacOS, and has so-far been tested using **Python 3.11**. We recommend using anaconda for package management, but pip should work as well.

1. Download and install both [ps_1loop](https://github.com/archaeo-pteryx/ps_1loop) and [ps_theory_calculator](https://github.com/archaeo-pteryx/ps_theory_calculator). You might need to request access to those repositories, in which case you can contact Yosuke Kobayashi (yosukekobayashi@arizona.edu). NOTE: This step will eventually be replaced with a more user-friendly process.
2. To enable GPU functionality for network training, make sure you have CUDA installed (or python 3.8+ if using apple arm64) and install the corresponding [PyTorch version](https://pytorch.org/get-started/locally/). If your machine doesn't have a GPU, you can skip this step.
3. Download this repository to your location of choice.
4. In the base directory, run `python -m pip install .`, which should install this repository as a package you can call anywhere on your machine.

To run the provided unit-tests, you can run the following command in the base repo directory,

`python -m pytest tests`


## Running the code

TODO: Update this section
