.. _install:

Installation Instructions
=========================

This package works on both Linux and MacOS (intel and arm64) platforms, and has so-far been tested using **Python 3.9 - 3.11**. There are two methods to install the code.

Preliminaries
-------------

1. To enable GPU functionality for network training, make sure you have CUDA installed (or python 3.8+ if using apple silicon).
2. You will need some way to generate galaxy power spectrum multipoles to generate training sets. One option is to download and install 
   both `ps_1loop`_ and `ps_theory_calculator`_. You might need to request access to those repositories, in which case you can contact Yosuke Kobayashi 
   (yosukekobayashi@arizona.edu). We have also included a version of `FAST-PT`_ to satisfy this requirnment.
3. Download this repository to your location of choice.

.. _ps_1loop: https://github.com/archaeo-pteryx/ps_1loop
.. _ps_theory_calculator: https://github.com/archaeo-pteryx/ps_theory_calculator
.. _FAST-PT: https://github.com/jablazek/FAST-PT

Automated Install (recommended)
-------------------------------

In the base directory, simply run ``install.sh`` in the terminal. This script will create a new anaconda enviornment, fetch the corresponding version of PyTorch, and install the code, all automatically.

Manual Install 
--------------

1. install the corresponding `PyTorch`_ version. If your machine doesn't have a GPU, you can skip this step.
2. In the base directory, run ``python -m pip install .``, which should install this repository as a package and all required dependencies.

.. _PyTorch: https://pytorch.org/get-started/locally/

To run the provided unit-tests, you can run the following command in the base repo directory,

``python -m pytest tests``