from spherex_emu.emulator import pk_emulator
import math, time, sys
import torch

def main():

    if len(sys.argv) < 2:
        print("USAGE: python train_emulator.py <config_file>")
        return 0

    t1 = time.time()
    emulator = pk_emulator(sys.argv[1])
    print("Training network on device:", emulator.device)
    emulator.train()
    t2 = time.time()
    print("Training Done in {:0.0f} hours {:0.0f} minutes {:0.2f} seconds\n".format(
        (t2-t1)//60, ((t2-t1) % 3600) // 60, (t2-t1)%60))

if __name__ == "__main__":
    main()
