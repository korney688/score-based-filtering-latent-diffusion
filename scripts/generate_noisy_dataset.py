import sys
import os
import gc
import logging
import h5py

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd()) 

from src.tools import set_seed, get_snr_batch
from src.plots import plot_snr_histogram
from src.input_repo.tools import get_noise, get_fixed_power_noise

# Инициализация Логгера
log = logging.getLogger(__name__)

def generate_noisy_dataset(cfg: DictConfig):
    
    # Считывание параметров
    try:
        device = 'cpu'
        seed = cfg.seed
        
        input_data_path = cfg.input_data_path
        
        output_dir = cfg.output_dir
        output_data_file = cfg.output_data_file
        output_data_path = cfg.output_data_path

        RW_scenario = cfg.AWGN_params.RW_scenario
        SNR_dB = cfg.AWGN_params.snr_db_range
        min_snr, max_snr = SNR_dB

        save_params = cfg.save_params

        log.info(f"Параметры успешно прочитаны")
        log.info(f"Selected Device: {device}")

        if not RW_scenario:
            log.info(f"min_snr: {min_snr}, max_snr: {max_snr}")
        else:
            log.info(f"Выбран Real-world scenario. Pn_sample_dB= -75 dB")
        
    except Exception as e:
        log.error(f'Ошибка при опредлении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e
        

    # Проверки доступности
    if os.path.exists(output_data_path):
        log.warning(f'Файл {output_data_path} уже существует.')
        log.warning("Остановка во избежание перезаписи.")
        return 
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    if (os.path.exists(input_data_path) == False):
        msg = f"Missing clean data at {input_data_path}"
        log.error(msg)
        raise FileNotFoundError(msg)

    # Настройка сидов
    set_seed(seed, device)
        
    # main loop
    log.info(f"Начало процесса зашумления")
    with h5py.File(input_data_path, 'r') as f_in, h5py.File(output_data_path, 'w') as f_out:
        
        # Входной датасет
        if 'dataset' not in f_in:
            raise ValueError("Ключ 'dataset' не найден в HDF5!")   
        input_dset = f_in['dataset']

        # Создаем выходной датасет
        n_samples = len(input_dset)
        sample_shape = input_dset.shape[1:]
        dtype_np = input_dset[0].dtype

        log.info(f"Start saving {n_samples} samples. Shape: {sample_shape}, Dtype: {dtype_np}, Noise: {min_snr} - {max_snr}")
        
        output_dset = f_out.create_dataset("dataset",
                                shape=(n_samples, *sample_shape), 
                                dtype='complex64',
                                compression="lzf",
                                chunks=(1, *sample_shape)
                                )
        
        output_snr = f_out.create_dataset("snr",
                                shape=(n_samples,), 
                                dtype='f4'
                                )

        batch_size = 1
        num_workers = 0

        dataset_loader = DataLoader(
            input_dset,                 
            batch_size=batch_size,
            shuffle=False,    
            drop_last=False,  
            num_workers=num_workers
        )
            
        current_idx = 0
        total_batches = len(dataset_loader)
        log_threshold = save_params.log_threshold

        pbar = tqdm(enumerate(dataset_loader), 
                    total=len(dataset_loader), 
                    desc='Запись датасета', 
                    ncols=80, mininterval=0.5,
                    leave=True)

        for i_batch, H_batch in pbar:
              
            # if not Real-world scenario
            if not RW_scenario:
                # sample normalisation
                H_batch = H_batch/torch.linalg.vector_norm(H_batch, dim=(1, 2, 3), keepdim=True)

                # get noise
                current_bs = H_batch.shape[0]
                current_snr = get_snr_batch(SNR_dB, current_bs)
                noise_batch = get_noise(H_batch, SNR_dB=current_snr)

                # add noise
                noisy_batch_data = (H_batch + noise_batch).detach().cpu().numpy()

            # if Real-world scenario
            else:
                # get noise
                current_bs = H_batch.shape[0]
                #fixed_snr = torch.tensor([-75] * current_bs) # fixed Pn_sample_dB=-75dB
                noise_batch = torch.cat([get_fixed_power_noise(shape = (4,64,96), Pn_sample_dB=-75) for _ in range(current_bs)])

                # add noise
                noisy_batch_data = (H_batch + noise_batch)

                # sample normalisation
                noisy_batch_data = noisy_batch_data/torch.linalg.vector_norm(noisy_batch_data, dim=(1, 2, 3), keepdim=True)
                noisy_batch_data = noisy_batch_data.detach().cpu().numpy()
                
            
            # Actual effective SNR, dB
            Ps = H_batch.abs().pow(2).mean((-1, -2, -3))
            Pn = noise_batch.abs().pow(2).mean((-1, -2, -3))
            actual_snr = 10 * torch.log10(Ps / Pn).numpy()

            # record
            output_dset[current_idx : current_idx + current_bs] = noisy_batch_data
            output_snr[current_idx : current_idx + current_bs] = actual_snr
            
            current_idx += current_bs

            progress_percent = (i_batch + 1) / total_batches * 100
            if progress_percent >= log_threshold:
                log.info(f"Обработано {int(log_threshold)}% данных")
                log_threshold += save_params.log_threshold

    log.info(f"Преобразование и сохранение завершено")

    # Тестирование. Сохранение распределения
    with h5py.File(output_data_path, 'r') as f_out:
        log.info(f"Построение распределения SNR в данных")
        
        # Входной датасет
        if 'dataset' not in f_out:
            raise ValueError("Ключ 'dataset' не найден в HDF5!") 
        if 'snr' not in f_out:
            raise ValueError("Ключ 'snr' не найден в HDF5!") 
        snr = f_out['snr'][:]
        
        filepath = os.path.join(output_dir, 'snr_distribution.png')
        plot_snr_histogram(data1=snr, filepath=filepath)
        log.info(f"Распределение SNR в данных успешно сохранено: {filepath}")