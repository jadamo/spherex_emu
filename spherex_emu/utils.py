from easydict import EasyDict
import yaml


def load_config_file(config_file:str):
    """loads in the emulator config file as a dictionary object
    
    Args:
        config_file: Config file path and name to laod
    """
    with open(config_file, "r") as stream:
        try:
            config_dict = EasyDict(yaml.safe_load(stream))
        except:
            print("ERROR! Couldn't read yaml file")
            return None
        
    # some basic checks that your config file has the correct formating    
    if len(config_dict.mlp_dims) != config_dict.num_mlp_blocks + 1:
        print("ERROR! mlp dimensions not formatted correctly!")
        return None
    if len(config_dict.parameter_bounds) != config_dict.input_dim:
        print("ERROR! parameter bounds not formatted correctly!")
        return None
    
    return config_dict