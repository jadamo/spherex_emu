import torch
import time
import itertools
import logging
import os

from spherex_emu.emulator import pk_emulator, compile_multiple_device_training_results
from spherex_emu.utils import calc_avg_loss, mse_loss, normalize_cosmo_params, pca_inverse_transform


def train_galaxy_ps_one_epoch(emulator:pk_emulator, train_loader:torch.utils.data.DataLoader, bin_idx:list):
    """Runs through one epoch of training for one sub-network in the galaxy_ps model

    Args:
        emulator (pk_emulator): emulator object to train
        train_loader (torch.utils.data.DataLoader): training data to loop through
        bin_idx (list): bin index [ps, z] identifying the sub-network to train.

    Returns:
        avg_loss (torch.Tensor): Average training-set loss. Used for backwards propagation
    """
    total_loss = 0.
    total_time = 0
    ps_idx = bin_idx[0]
    z_idx  = bin_idx[1]
    net_idx = (z_idx * emulator.num_spectra) + ps_idx
    for (i, batch) in enumerate(train_loader):
        t1 = time.time()
        
        # setup input parameters
        params = emulator.galaxy_ps_model.organize_parameters(batch[0])
        params = normalize_cosmo_params(params, emulator.input_normalizations)
        
        target = torch.flatten(batch[1][:,ps_idx,z_idx], start_dim=1)
        prediction = emulator.galaxy_ps_model.forward(params, net_idx)
        
        # calculate loss and update network parameters
        # TODO: Find better way to deal with passing invcov to the loss function
        if emulator.normalization_type == "pca":
            prediction = pca_inverse_transform(prediction, emulator.principle_components, emulator.training_set_variance)
            loss = emulator.loss_function(prediction, target, emulator.invcov_blocks[ps_idx, z_idx], False)
        else:
            loss = emulator.loss_function(prediction, target, emulator.invcov_full, True)

        assert torch.isnan(loss) == False
        assert torch.isinf(loss) == False
        emulator.optimizer[ps_idx][z_idx].zero_grad(set_to_none=True)
        loss.backward()

        emulator.optimizer[ps_idx][z_idx].step()
        total_loss += loss.detach()
        total_time += (time.time() - t1)

    emulator.logger.debug("time for epoch: {:0.1f}s, time per batch: {:0.1f}ms".format(total_time, 1000*total_time / len(train_loader)))
    return (total_loss / len(train_loader.dataset))


def train_nw_ps_one_epoch(emulator:pk_emulator, train_loader):
    """Runs through one epoch of training for the non-wiggle linear power spectrum model.
    NOTE: This is experimental and not currently used!

    Args:
        emulator (pk_emulator): emulator object to train
        train_loader (torch.utils.data.DataLoader): training data to loop through

    Returns:
        avg_loss (torch.Tensor): Average training-set loss. Used for backwards propagation
    """
    total_loss = 0.
    total_time = 0
    for (i, batch) in enumerate(train_loader):
        t1 = time.time()
        
        # setup input parameters
        params = batch[0][:,:emulator.num_cosmo_params]
        params = normalize_cosmo_params(params, emulator.input_normalizations[:,0,:emulator.num_cosmo_params])

        target = batch[2]
        prediction = emulator.nw_ps_model.forward(params)

        # calculate loss and update network parameters
        # TODO: Find better way to deal with passing invcov to the loss function
        loss = mse_loss(prediction, target, emulator.invcov_full, True)
        assert torch.isnan(loss) == False
        assert torch.isinf(loss) == False
        emulator.nw_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(emulator.nw_ps_model.parameters(), 1e8)    
        emulator.nw_optimizer.step()
        total_loss += loss.detach()
        total_time += (time.time() - t1)

    emulator.logger.debug("time for epoch: {:0.1f}s, time per batch: {:0.1f}ms".format(total_time, 1000*total_time / len(train_loader)))
    return (total_loss / len(train_loader.dataset))


def train_on_single_device(emulator:pk_emulator):
    """Trains the emulator on a single device (cpu or gpu)

    Args:
        emulator (pk_emulator): network object to train.
    """

    # load training / validation datasets
    train_loader = emulator.load_data("training", emulator.training_set_fraction)
    valid_loader = emulator.load_data("validation")

    best_loss           = [torch.inf for i in range(emulator.num_zbins*emulator.num_spectra + 1)]
    epochs_since_update = [0 for i in range(emulator.num_zbins*emulator.num_spectra + 1)]
    emulator._init_training_stats()
    emulator._init_optimizer()

    emulator.galaxy_ps_model.train()
    emulator.nw_ps_model.train()

    start_time = time.time()
    # loop thru epochs
    for epoch in range(emulator.num_epochs):

        # loop thru individual networks
        for (ps, z) in itertools.product(range(emulator.num_spectra), range(emulator.num_zbins)):
            net_idx = (z * emulator.num_spectra) + ps
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                continue

            training_loss = train_galaxy_ps_one_epoch(emulator, train_loader, [ps, z])
            if emulator.recalculate_train_loss:
                emulator.train_loss[ps][z].append(calc_avg_loss(emulator, train_loader, emulator.loss_function, [ps, z], "galaxy_ps"))
            else:
                emulator.train_loss[ps][z].append(training_loss)
            emulator.valid_loss[ps][z].append(calc_avg_loss(emulator, valid_loader, emulator.loss_function, [ps, z], "galaxy_ps"))
            
            emulator.scheduler[ps][z].step(emulator.valid_loss[ps][z][-1])
            emulator.train_time = time.time() - start_time

            if emulator.valid_loss[ps][z][-1] < best_loss[net_idx]:
                best_loss[net_idx] = emulator.valid_loss[ps][z][-1]
                epochs_since_update[net_idx] = 0
                emulator._update_checkpoint(net_idx, "galaxy_ps")
            else:
                epochs_since_update[net_idx] += 1

            emulator.logger.info("Net idx : [{:d}, {:d}], Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(
                ps, z, epoch, emulator.train_loss[ps][z][-1], emulator.valid_loss[ps][z][-1], epochs_since_update[net_idx]))
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                emulator.logger.info("Model [{:d}, {:d}] has not impvored for {:0.0f} epochs. Initiating early stopping...".format(ps, z, epochs_since_update[net_idx]))

        # # train non-wiggle power spectrum network
        # if epochs_since_update[-1] > emulator.early_stopping_epochs:
        #     continue

        # emulator.nw_train_loss.append(train_nw_ps_one_epoch(emulator, train_loader))
        # emulator.nw_valid_loss.append(calc_avg_loss(emulator.nw_ps_model, valid_loader, emulator.input_normalizations, 
        #                                             emulator.invcov_full, mse_loss, [ps, z], "nw_ps"))
        
        # emulator.nw_scheduler.step(emulator.nw_valid_loss[-1])
        # emulator.train_time = time.time() - start_time

        # if emulator.nw_valid_loss[-1] < best_loss[-1]:
        #     best_loss[-1] = emulator.nw_valid_loss[-1]
        #     epochs_since_update[-1] = 0
        #     emulator._update_checkpoint(mode="nw_ps")
        # else:
        #     epochs_since_update[-1] += 1

        # if emulator.print_progress: print("Non-wiggle net, Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(
        #     epoch, emulator.nw_train_loss[-1], emulator.nw_valid_loss[-1], epochs_since_update[-1]), flush=True)
        # if epochs_since_update[-1] > emulator.early_stopping_epochs:
        #     print("Non-wiggle net has not impvored for {:0.0f} epochs. Initiating early stopping...".format(epochs_since_update[-1]), flush=True)


def train_on_multiple_devices(gpu_id:int, net_indeces:list, config_dir:str):
    """Trains the given network on multiple gpu devices by splitting.

    This function is called in parralel using multiproccesing, and works by training specific sub-networks 
    on seperate gpus, each saving to a seperate sub-directory. After 25 epochs have passed on gpu 0, the results from all gpus are compiles together
    and saved in the base save directory

    Args:
        gpu_id (int): gpu number for logging and organizing save location.
        net_indeces (list): List of sub-network indices to train on the given gpu. This is different for each gpu
        config_dir (str): Location of the input network config file.
    """
    # Each sub-process gets its own indpendent emulator object, where it will train the corresponding
    # sub-networks based on net_indeces
    device = torch.device(f"cuda:{gpu_id}")
    logging.basicConfig(level=logging.DEBUG, format=f"[GPU {gpu_id}] %(message)s")
    emulator = pk_emulator(config_dir, "train", device)

    base_save_dir = os.path.join(emulator.input_dir, emulator.save_dir)
    emulator.save_dir += "rank_"+str(gpu_id)+"/"
    emulator.logger.debug(f"training networks with ids: {net_indeces[gpu_id]}")

    train_loader = emulator.load_data("training", emulator.training_set_fraction)
    valid_loader = emulator.load_data("validation")

    best_loss           = [torch.inf for i in range(emulator.num_zbins*emulator.num_spectra + 1)]
    epochs_since_update = [0 for i in range(emulator.num_zbins*emulator.num_spectra + 1)]
    emulator._init_training_stats()
    emulator._init_optimizer()

    emulator.galaxy_ps_model.train()
    emulator.nw_ps_model.train()

    start_time = time.time()
    # loop thru epochs
    for epoch in range(emulator.num_epochs):
        # loop thru individual networks
        for (ps, z) in net_indeces[gpu_id]:
            net_idx = int((z * emulator.num_spectra) + ps)
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                continue

            training_loss = train_galaxy_ps_one_epoch(emulator, train_loader, [ps, z])
            if emulator.recalculate_train_loss:
                emulator.train_loss[ps][z].append(calc_avg_loss(emulator, train_loader, emulator.loss_function, [ps, z], "galaxy_ps"))
            else:
                emulator.train_loss[ps][z].append(training_loss)
            emulator.valid_loss[ps][z].append(calc_avg_loss(emulator, valid_loader, emulator.loss_function, [ps, z], "galaxy_ps"))
            
            emulator.scheduler[ps][z].step(emulator.valid_loss[ps][z][-1])
            emulator.train_time = time.time() - start_time

            if emulator.valid_loss[ps][z][-1] < best_loss[net_idx]:
                best_loss[net_idx] = emulator.valid_loss[ps][z][-1]
                epochs_since_update[net_idx] = 0
                emulator._update_checkpoint(net_idx, "galaxy_ps")
            else:
                epochs_since_update[net_idx] += 1

            emulator.logger.info("Net idx : [{:d}, {:d}], Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(
                ps, z, epoch, emulator.train_loss[ps][z][-1], emulator.valid_loss[ps][z][-1], epochs_since_update[net_idx]))
            if epochs_since_update[net_idx] > emulator.early_stopping_epochs:
                emulator.logger.info("Model [{:d}, {:d}] has not impvored for {:0.0f} epochs. Initiating early stopping...".format(ps, z, epochs_since_update[net_idx]))

        # train non-wiggle power spectrum network        
        # if gpu_id == 0 and epochs_since_update[-1] <= emulator.early_stopping_epochs:
        #     emulator.nw_train_loss.append(train_nw_ps_one_epoch(emulator, train_loader))
        #     emulator.nw_valid_loss.append(calc_avg_loss(emulator, valid_loader, mse_loss, [ps, z], "nw_ps"))
            
        #     emulator.nw_scheduler.step(emulator.nw_valid_loss[-1])
        #     emulator.train_time = time.time() - start_time

        #     if emulator.nw_valid_loss[-1] < best_loss[-1]:
        #         best_loss[-1] = emulator.nw_valid_loss[-1]
        #         epochs_since_update[-1] = 0
        #         emulator._update_checkpoint(mode="nw_ps")
        #     else:
        #         epochs_since_update[-1] += 1

        #     if emulator.print_progress: print("Non-wiggle net, Epoch : {:d}, avg train loss: {:0.4e}\t avg validation loss: {:0.4e}\t ({:0.0f})".format(
        #         epoch, emulator.nw_train_loss[-1], emulator.nw_valid_loss[-1], epochs_since_update[-1]), flush=True)
        #     if epochs_since_update[-1] > emulator.early_stopping_epochs:
        #         print("Non-wiggle net has not impvored for {:0.0f} epochs. Initiating early stopping...".format(epochs_since_update[-1]), flush=True)

        if gpu_id == 0 and epoch % 5 == 0 and epoch > 0:
            emulator.logger.info("Checkpointing progress from all devices...")
            full_emulator = compile_multiple_device_training_results(base_save_dir, config_dir, emulator.num_gpus)
            full_emulator._save_model()
