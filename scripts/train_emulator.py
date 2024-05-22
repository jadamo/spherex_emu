from spherex_emu.emulator import pk_emulator

def main():

    config_file = "/home/joeadamo/Research/SPHEREx/spherex_emu/configs/example.yaml"

    emulator = pk_emulator(config_file)
    emulator.train()

if __name__ == "__main__":
    main()