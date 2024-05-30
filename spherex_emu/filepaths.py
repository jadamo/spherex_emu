# -----------------------------------------------
# this file contains all necesary file paths to use this repository
import os

# Directory with neural network config files
# by default this is the "configs" directory within spherex_emu itself
network_pars_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/configs/network_pars/"

# Directory with survey parameter files
survey_pars_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/configs/survey_pars/"

# Directory with cosmologial parameter files
cosmo_pars_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/configs/cosmo_pars/"