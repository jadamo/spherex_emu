import torch
import time
import itertools

from spherex_emu.emulator import pk_emulator, compile_multiple_device_training_results
from spherex_emu.utils import calc_avg_loss

# TODO: Move other training loop here AFTER Grace is done

def train_on_multiple_devices(gpu_id, net_indeces, config_dir):
    
    device = torch.device(f"cuda:{gpu_id}")
    # Each sub-process gets its own indpendent emulator object, where it will train the corresponding
    # sub-networks based on net_indeces
    emulator = pk_emulator(config_dir, "train", device)
    base_save_dir = emulator.save_dir
    emulator.save_dir += "rank_"+str(gpu_id)+"/"
    print(device, net_indeces[gpu_id], flush=True)

    train_loader = emulator.load_data("training", emulator.training_set_fraction)
    valid_loader = emulator.load_data("validation")

    # store training data as nested lists with dims [nps, nz]
    emulator.train_loss     = [[[] for i in range(emulator.num_zbins)] for j in range(emulator.num_spectra)]
    emulator.valid_loss     = [[[] for i in range(emulator.num_zbins)] for j in range(emulator.num_spectra)]
    best_loss           = [[torch.inf for i in range(emulator.num_zbins)] for j in range(emulator.num_spectra)]
    epochs_since_update = [[0 for i in range(emulator.num_zbins)] for j in range(emulator.num_spectra)]
    emulator.train_time = 0.
    if emulator.print_progress: print("Initial learning rate = {:0.2e}".format(emulator.learning_rate), flush=True)
    
    emulator._set_optimizer()
    emulator.model.train()

    start_time = time.time()
    # loop thru epochs
    for epoch in range(emulator.num_epochs):
        # loop thru individual networks
        for (ps, z) in net_indeces[gpu_id]:
            if epochs_since_update[ps][z] > emulator.early_stopping_epochs:
                continue

            training_loss = emulator._train_one_epoch(train_loader, [ps, z])
            if emulator.recalculate_train_loss:
                emulator.train_loss[ps][z].append(calc_avg_loss(emulator.model, train_loader, emulator.nput_normalizations, 
                                                            emulator.ps_fid, emulator.invcov_full, emulator.sqrt_eigvals, 
                                                            emulator.Q, emulator.Q_inv, emulator.loss_function, [ps, z]))
            else:
                emulator.train_loss[ps][z].append(training_loss)
            emulator.valid_loss[ps][z].append(calc_avg_loss(emulator.model, valid_loader, emulator.input_normalizations, 
                                                        emulator.ps_fid, emulator.invcov_full, emulator.sqrt_eigvals, 
                                                        emulator.Q, emulator.Q_inv, emulator.loss_function, [ps, z]))
            
            emulator.scheduler[ps][z].step(emulator.valid_loss[ps][z][-1])
            emulator.train_time = time.time() - start_time

            if emulator.valid_loss[ps][z][-1] < best_loss[ps][z]:
                best_loss[ps][z] = emulator.valid_loss[ps][z][-1]
                epochs_since_update[ps][z] = 0
                emulator._save_model()
            else:
                epochs_since_update[ps][z] += 1

            if emulator.print_progress: print("GPU: {:d}, Net idx : [{:d}, {:d}], Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(
                gpu_id, ps, z, epoch, emulator.train_loss[ps][z][-1], emulator.valid_loss[ps][z][-1], epochs_since_update[ps][z]), flush=True)
            if epochs_since_update[ps][z] > emulator.early_stopping_epochs:
                print("Model [{:d}, {:d}] has not impvored for {:0.0f} epochs. Initiating early stopping...".format(ps, z, epochs_since_update[ps][z]), flush=True)

        if gpu_id == 0 and epoch % 50 == 0 and epoch > 0:
            print("Checkpointing progress from all devices...", flush=True)
            full_emulator = compile_multiple_device_training_results(emulator.input_dir + base_save_dir, config_dir, emulator.num_gpus)
            full_emulator._save_model()