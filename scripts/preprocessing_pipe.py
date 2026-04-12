import sys
import os
import gc
import logging
from glob import glob

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import h5py
import random


sys.path.append(os.getcwd()) 

from src.tools import set_seed
from src.datasets import my_dataset

from src.input_repo.dataset import (
    MultiRoadDataset,
    SubbandSelect,
    Freq2Delay,
    HardWindow,
    Antenna2Beam,
)


log = logging.getLogger(__name__)


def get_dataset(dir_path, params):
    
    file_pattern = '*.mat'

    # Считываем директорию с файлами
    pathlist = glob(os.path.join(dir_path, file_pattern))
    
    if not pathlist:
        msg = f"Внимание: Файлы не найдены в {dir_path}"
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info(f"Найдено файлов: {len(pathlist)}")

    # Параметры препроцессинга
    N_subbands = params.SubbandSelect.N_subbands
    method = params.SubbandSelect.method
    N_pol = params.Antenna2Beam.N_pol
    N_hor = params.Antenna2Beam.N_hor
    N_ver = params.Antenna2Beam.N_ver
    
    transform_pipe = transforms.Compose(
        [
            SubbandSelect(N_subbands=N_subbands, method=method),
            Freq2Delay(inverse=False),
            Antenna2Beam(N_pol=N_pol, N_hor=N_hor, N_ver=N_ver, inverse=False),
        ]
    )

    dataset = MultiRoadDataset(paths=pathlist, transform=transform_pipe)

    # Проверка параметров датасета
    n_samples = len(dataset)
    if n_samples == 0:
        msg = "Датасет пуст."
        log.error(msg)
        raise ValueError(msg) 

    batch_shape = dataset[0].shape
    full_data_shape = (n_samples, *batch_shape)
    log.info(f"Data Shape: {full_data_shape}")

    return dataset

def save_h5dataset(output_path, dataset, params):

    try:
        with h5py.File(output_path, 'w') as f:
    
            n_samples = len(dataset)
            first_sample = dataset[0]
            sample_shape = first_sample.shape
            dtype_np = first_sample.numpy().dtype
    
            log.info(f"Start saving {n_samples} samples. Shape: {sample_shape}, Dtype: {dtype_np}")
            
            # Инициируем датасет
            dset = f.create_dataset("dataset",
                                    shape=(n_samples, *sample_shape), 
                                    dtype='complex64',
                                    compression="lzf",
                                    chunks=(1, *sample_shape)
                                    )
        
            # Инициализация Dataloader
            batch_size = params.batch_size
            num_workers = 0
            
            dataset_loader = DataLoader(
                dataset,                 
                batch_size=batch_size,   
                shuffle=False,    
                drop_last=False,  
                num_workers=num_workers    
            )
        
            
            # Параметры логирования
            #log.info(f"Начало преобразования")
            total_batches = len(dataset_loader)
            log_threshold = params.log_threshold
        
            
            # Инициируеем tqdm
            pbar = tqdm(enumerate(dataset_loader), 
                        total=len(dataset_loader), 
                        desc='Запись датасета', 
                        ncols=80, mininterval=0.5,
                        leave=True)
        
            
            current_idx = 0
            for i_batch, H_batch in pbar:
                
                batch_data = H_batch.detach().cpu().numpy()
                current_bs = H_batch.shape[0]
                
                end_idx = current_idx + current_bs
                dset[current_idx : end_idx] = batch_data
                
                current_idx += current_bs
        
                progress_percent = (i_batch + 1) / total_batches * 100
                if progress_percent >= log_threshold:
                    log.info(f"Обработано {int(log_threshold)}% данных (батч {i_batch + 1}/{total_batches})")
                    log_threshold += params.log_threshold

        log.info(f"Преобразование и сохранение завершено")
    
    except Exception as e:
        log.error(f"Ошибка при сохранении датасета: {e}")


def tests(h5_path, original_dataset, num_checks=5):
    """
    Быстрая проверка целостности записанного HDF5.
    """
    
    log.info(f"Запуск верификации данных: {h5_path}")
    
    dset = my_dataset(
        h5_path=h5_path,
        data_key="dataset",
        in_memory=False,
        apply_log=False,
        apply_norm=False,
        apply_split=False
        )
    
    # 1. Проверка длины
    if len(dset) != len(original_dataset):
        raise ValueError(f"Размер не совпадает! H5: {len(dset)}, Orig: {len(original_dataset)}")

    # 2. Проверка шейпа (без учета первой оси N)
    orig_sample = original_dataset[0]
    
    if dset.shape[1:] != orig_sample.shape:
         raise ValueError(f"Shape сэмпла не совпадает! H5: {dset.shape[1:]}, Orig: {orig_sample.shape}")

    # 3. Выборочная проверка значений
    # Берем индексы: первый, последний и несколько случайных
    indices = [0, len(dset)-1] 
    indices += random.sample(range(1, len(dset)-1), min(num_checks, len(dset)-2))
        
    for idx in indices:
        
        h5_data = dset[idx]
        
        orig_data = original_dataset[idx]
        if hasattr(orig_data, 'numpy'):
            orig_data = orig_data.numpy()

        # complex
        if not np.allclose(h5_data, orig_data, rtol=1e-5):
            raise ValueError(f"Данные не совпадают на индексе {idx}!\n"
                             f"Max diff: {np.max(np.abs(h5_data - orig_data))}")
    
    log.info("Верификация успешна. Данные идентичны.")

def preprocessing_pipe(cfg: DictConfig):

    try:
        device = 'cpu'
        seed = cfg.seed
        
        raw_data_dir = to_absolute_path(cfg.input_dir)
        
        output_data_dir = cfg.output_dir
        tr_params = cfg.transforms_params
        save_params = cfg.save_params
    except Exception as e:
        log.error(f'Ошибка при опредлении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e

    log.info(f"Параметры успешно прочитаны")
    log.info(f"Selected Device: {device}")

    # Создание директории и контроль перезаписи
    output_file = 'dataset_clean.h5'  
    output_path = os.path.join(output_data_dir, output_file)
    if os.path.exists(output_path):
        log.error(f'Файл {output_path} уже существует -> ПРОВЕРКА -> ОСТАНОВКА')
        
        # Получение датасета (включая препроцессинг)
        dataset = get_dataset(raw_data_dir, tr_params)
        # Верификация
        tests(output_path, dataset, num_checks=10)
        return
    else:
        os.makedirs(output_data_dir, exist_ok=True)

    # Настройка сидов
    set_seed(seed, device)

    # Получение датасета (включая препроцессинг)
    dataset = get_dataset(raw_data_dir, tr_params)

    # Сохранение в виде .h5 файла
    save_h5dataset(output_path, dataset, save_params)
    
    # Верификация
    tests(output_path, dataset, num_checks=10)