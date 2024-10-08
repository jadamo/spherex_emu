#-----------------------------------------------
# this file contains all necesary file paths to use this repository
import os

# repository directory
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/"

# Directory with neural network config files
# by default this is the "configs" directory within spherex_emu itself
network_pars_dir = base_dir+"configs/network_pars/"

# Directory with survey parameter files
survey_pars_dir = base_dir+"configs/survey_pars/"

# Directory with cosmologial parameter files
cosmo_pars_dir = base_dir+"configs/cosmo_pars/"

data_dir = base_dir+"data/"
