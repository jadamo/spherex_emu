from spherex_emu.emulator import pk_emulator
from spherex_emu.filepaths import network_pars_dir
from spherex_emu.utils import load_config_file
import math, time, torch
import numpy as np

def main():

    config_file = network_pars_dir + "network_pars_single_tracer_single_redshift.yaml"
    config_dict = load_config_file(config_file)

    batch_size = [100, 200, 300, 400, 500]
    learning_rates = np.logspace(-4, -2, 10)
    best_loss = torch.zeros((len(batch_size), len(learning_rates), 2, 2))

    for batch in range(len(batch_size)):
        for lr in range(len(learning_rates)):
            
            config_dict["batch_size"] = batch_size[batch]
            config_dict["learning_rate"][0] = float(learning_rates[lr])

            for round in range(2):
                emulator = pk_emulator(config_dict=config_dict)
                emulator.train(print_progress=True)

                idx = torch.argmin(torch.tensor(emulator.valid_loss)).item()
                print(idx)
                best_loss[batch, lr, round, 0] = emulator.train_loss[idx]
                best_loss[batch, lr, round, 1] = emulator.valid_loss[idx]

                print("batch size = {:0.0f}, lr = {:0.2e}, best loss = [{:0.3e}, {:0.3e}]".format(
                       batch_size[batch], learning_rates[lr], best_loss[batch, lr, round, 0], best_loss[batch, lr, round, 1]))
                torch.save(best_loss, emulator.save_dir + "optimization-data.dat")

    t1 = time.time()
    emulator = pk_emulator(config_file)
    emulator.train()
    t2 = time.time()
    print("Training Done in took {:0.0f} minutes {:0.2f} seconds\n".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()