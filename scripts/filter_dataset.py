import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import logging
import h5py

import sys
import os
import shutil
import time
import datetime

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd()) 

# Импорт пользовательских функций
from src.datasets import my_dataset
from src.tools import set_seed
from src.DDPM_model import build_DDPM_model
from src.filters import run_DDPM_filter, run_DDPM_filter_v2


# Инициализация Логгера
log = logging.getLogger(__name__)

def data_filtering (cfg: DictConfig):

    # ------------------------------------------------------------------
    # Загрузка конфигурации
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed
        
        dataset_path = cfg.dataset_path
        output_dir = cfg.output_dir
        DDPM_model_path = cfg.DDPM_model_path
        
        percent = cfg.save_percent
        mode = cfg.filter_mode

        # Модификация
        ver = cfg.get('ver', 'v1')
        params = cfg.get('params', [10,20,10])
        
        in_memory = cfg.get('in_memory', 'False')
        
        log.info(f"Параметры успешно считаны. device: {device}")
    except Exception as e:
        log.error(f'Ошибка при определении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e
    
    log.info(f"Start filtering_dataset")

    # ------------------------------------------------------------------
    # Проверка путей
    log.info(f"Провекра наличия предобученной модели DDPM")
    if not os.path.exists(DDPM_model_path):
        log.error(f"Чекпоинт модели НЕ найден: {DDPM_model_path}")
        raise FileNotFoundError(f"Model checkpoint missing: {DDPM_model_path}")

    log.info(f"Чекпоинт найден: {DDPM_model_path}")
    
    DDPM_checkpoint = torch.load(DDPM_model_path, map_location=device)
        

    # ------------------------------------------------------------------
    # Инициализация датасета
    DDPM_params = DDPM_checkpoint['DDPM_params']
    
    dataset_norm = my_dataset(
        h5_path=dataset_path,
        data_key="dataset",
        in_memory=in_memory,
        apply_log=DDPM_params['apply_log'],
        apply_norm=DDPM_params['apply_norm'],
        apply_split=DDPM_params['apply_split'],
        data_mode="image"
        )

    # ------------------------------------------------------------------
    # Загрузка параметров модели
    try:
        base_dim = DDPM_params['base_dim']
        deep = DDPM_params.get('deep', 3)
        DDPM_model = build_DDPM_model(base_dim, deep, device)
        DDPM_model.model.load_state_dict(DDPM_checkpoint['model_state_dict'])
        DDPM_model.model.eval()
        
        log.info("Модель DDPM загружена и переведена в режим eval.")
    except Exception as e:
        log.error(f"Ошибка при инициализации модели: {e}")
        raise e

    # ------------------------------------------------------------------
    # score_base filtering
    if ver=='v2':
        indices = run_DDPM_filter_v2(dataset_norm, DDPM_model, device, mode, percent, params)
    else:
        indices = run_DDPM_filter(dataset_norm, DDPM_model, device, mode, percent)
    log.info(f"Процедура data_filtering завершена. Возвращено {len(indices)} индексов.")

    return indices



# Основная функция
def filter_dataset(cfg: DictConfig):

    # ------------------------------------------------------------------
    # Загрузка конфигурации
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed
        
        dataset_path = cfg.dataset_path
        output_dir = cfg.output_dir
    
        filter_mode = cfg.filter_mode
        save_percent = cfg.save_percent

        in_memory = cfg.get('in_memory', 'False')
        
        log.info("Параметры успешно считаны")
    except Exception as e:
        log.error(f'Ошибка при определении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e
    
    log.info(f"Start filtering_dataset")

    # ------------------------------------------------------------------
    # Проверка путей
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_dir) and any(os.scandir(output_dir)):
        msg = f"Output directory is not empty: {output_dir}"
        log.error(msg)
        raise FileExistsError(msg)
    
    # Проверка доступности входных данных
    if not os.path.exists(dataset_path):
        log.error(f"Missing clean data at {dataset_path}")
        raise FileNotFoundError(f"Missing clean data at {dataset_path}")

    # ------------------------------------------------------------------
    # Настройка сидов
    set_seed(seed, device)

    # ------------------------------------------------------------------
    # Инициализация датасета
    full_dataset = my_dataset(
        h5_path=dataset_path,
        data_key="dataset",
        snr_key='snr',
        in_memory=in_memory,
        apply_log=False,
        apply_norm=False,
        apply_split=False,
        data_mode="image"
        )
    
    log.info(f'Dataset shape: {full_dataset.shape}')

    # ------------------------------------------------------------------
    # Фильтрация датасета
    if filter_mode == 'random':
        log.info(f"Начало отбора данных. Метод: {filter_mode}, Keep: {save_percent}%")
        indices = np.random.choice(full_dataset.shape[0], size=int(full_dataset.shape[0] * save_percent/100), replace=False)
    else:
        indices = data_filtering(cfg)
   
    # ------------------------------------------------------------------
    # Сохранение отфильтрованного датасета
    filtered_subset = Subset(full_dataset, indices)
    log.info(f'Filtered dataset shape: {len(indices)}')
    
    batch_size = 32 # или больше, зависит от вашей RAM
    filtered_loader = DataLoader(filtered_subset, batch_size=batch_size, shuffle=False)

    save_path = os.path.join(output_dir, "filtered_dataset.h5")
    try:
        with h5py.File(save_path, 'w') as hf:
            # 1. Берем первый элемент для определения размерностей и ТИПА ДАННЫХ
            sample_data, sample_snr = full_dataset[0]
            
            # Определяем форму (shape)
            data_shape = (len(indices), *sample_data.shape)
            snr_shape = (len(indices),) if getattr(sample_snr, 'shape', None) in [(), None] else (len(indices), *sample_snr.shape)
            
            # 2. ДИНАМИЧЕСКИ извлекаем оригинальный тип (dtype), конвертируя семпл в numpy
            data_dtype = sample_data.cpu().numpy().dtype if isinstance(sample_data, torch.Tensor) else np.array(sample_data).dtype
            snr_dtype = sample_snr.cpu().numpy().dtype if isinstance(sample_snr, torch.Tensor) else np.array(sample_snr).dtype
            
            log.info(f"Определены типы данных для сохранения. Dataset: {data_dtype}, SNR: {snr_dtype}")

            # 3. Создаем HDF5 с ОРИГИНАЛЬНЫМИ типами (вместо жесткого np.float32)
            h5_dataset = hf.create_dataset("dataset", shape=data_shape, dtype=data_dtype)
            h5_snr = hf.create_dataset("snr", shape=snr_shape, dtype=snr_dtype)
            
            start_idx = 0
            # Сохраняем по батчам
            for batch_data, batch_snr in tqdm(filtered_loader, desc="Saving H5"):
                end_idx = start_idx + batch_data.shape[0]
                
                # Отвязываем от графов и переводим в numpy
                np_data = batch_data.detach().cpu().numpy()
                np_snr = batch_snr.detach().cpu().numpy()
                
                # Принудительно страхуем совпадение типов с HDF5 
                # (на случай, если Dataloader где-то кастанул тип)
                np_data = np_data.astype(data_dtype)
                np_snr = np_snr.astype(snr_dtype)
                
                h5_dataset[start_idx:end_idx] = np_data
                h5_snr[start_idx:end_idx] = np_snr
                
                start_idx = end_idx
                
        log.info(f"Успешно сохранено: {save_path}")
        
    except Exception as e:
        log.error(f"Ошибка при сохранении .h5 файла: {e}")
        raise e
