from spherex_emu.emulator import pk_emulator
from spherex_emu.filepaths import network_pars_dir
import math, time

def main():

    config_file = network_pars_dir + "network_pars_2_sample_2_redshift.yaml"

    t1 = time.time()
    emulator = pk_emulator(config_file)
    emulator.train()
    t2 = time.time()
    print("Training Done in took {:0.0f} minutes {:0.2f} seconds\n".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()