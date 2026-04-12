from glob import glob
import logging
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset, SequentialSampler
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
import re

from src.input_repo.dataset import (
    MultiRoadDataset,
    SubbandSelect,
    Freq2Delay,
    HardWindow,
    Antenna2Beam
)

from src.datasets import my_dataset
from src.input_repo.tools import get_noise, get_fixed_power_noise
from src.input_repo.tools import get_precoder, capacity
from src.input_repo.TDnCNN import build_DnCNN_3D_model

log = logging.getLogger(__name__)


def get_snr_batch(SNR_dB, size):
    """Функция формирования SNR под батч"""
    
    if np.isscalar(SNR_dB):
        return np.full(size, SNR_dB)
        
    low, high = sorted(SNR_dB)
    
    if low == high:
        high = high + 1

    np.random.seed(42)
    return np.random.randint(low=low, high=high, size=size)


def read_checkpoint(path_abs, device):
    TDnCNN_checkpoint = torch.load(path_abs, map_location=device)
    model_size = TDnCNN_checkpoint['model_size']
    epoch_n = TDnCNN_checkpoint['epoch']
    backward_n = TDnCNN_checkpoint['backward_n']
    val_loss = TDnCNN_checkpoint['val_loss']
    TDnCNN_model = build_DnCNN_3D_model(device, model_size)
    TDnCNN_model.load_state_dict(TDnCNN_checkpoint['model_state_dict'])

    return TDnCNN_model.eval(), epoch_n, backward_n, val_loss

def get_epoch_number(file_path):

    pattern = r'epoch_(\d+)\.pth$'
    match = re.search(pattern, file_path)
    
    if match:
        epoch_number = int(match.group(1))
        return epoch_number
    else:
        return file_path

def calc_gap(loader, TDnCNN_model, RW_scenario, UL_SNR_db, max_sim_rank, device, limit_batches):
    '''Функция расчета capacity и gap для ideal/denoised'''
    
    # Variables for storing the results
    capacity_true = np.zeros((0, max_sim_rank))
    capacity_denoised = np.zeros((0, max_sim_rank))
    
    DL_SNR_db = 30    

    # ------------------------------------------------------------------
    # Расчет capacity
    with torch.no_grad():
        for i_batch, H_batch in enumerate(loader):

            # ideal
            H_batch = H_batch.to(device)

            # add Noise
            # if not Real-world scenario
            if not RW_scenario:
    
                # sample normalisation (ранее была в MultiRoadDataset)
                H_batch = H_batch/torch.linalg.vector_norm(H_batch, dim=(1, 2, 3), keepdim=True)
    
                # генерация шума
                current_bs = H_batch.shape[0]
                current_snr = get_snr_batch(UL_SNR_db, current_bs)
                
                # Генерируем шум
                noise_batch = get_noise(H_batch, SNR_dB=current_snr)
                noise_batch = noise_batch.to(device)

                noisy_batch = (H_batch + noise_batch)

            # if Real-world scenario
            else:
                # get noise
                current_bs = H_batch.shape[0]
                #fixed_snr = torch.tensor([-75] * current_bs) # fixed Pn_sample_dB=-75dB
                noise_batch = torch.cat([get_fixed_power_noise(shape = (4,64,96), Pn_sample_dB=-75) for _ in range(current_bs)])
                noise_batch = noise_batch.to(device)

                noisy_batch = (H_batch + noise_batch)
                
                # sample normalisation. After add noise
                noisy_batch = noisy_batch/torch.linalg.vector_norm(noisy_batch, dim=(1, 2, 3), keepdim=True)
                H_batch = H_batch/torch.linalg.vector_norm(H_batch, dim=(1, 2, 3), keepdim=True)
            
            # denoising
            denoised_batch = TDnCNN_model(noisy_batch).detach()

            # Transform delay-domain into frequency
            delay2Freq = Freq2Delay(inverse=True)
        
            # to freq-domain channel
            H_freq_ideal = delay2Freq(H_batch)
            H_freq_denoised = delay2Freq(denoised_batch)
        
            # Calculate precoders for all ranks using ideal and noisy channels
            W_batch_ideal = get_precoder(H_freq_ideal, rank=max_sim_rank)
            W_batch_denoised = get_precoder(H_freq_denoised, rank=max_sim_rank)
        
            # Calculate capacity per rank
            # ideal
            C_buff = []
            for i_rank in range(1, max_sim_rank + 1):
                # select precoder of requested rank, Average capacity over subcariers
                mean_capacity = capacity(
                    H_freq_ideal, W_batch_ideal[..., :i_rank], SNR_dB=DL_SNR_db
                ).mean(-1)
                C_buff.append(mean_capacity.detach().cpu().numpy())
        
            # concatenate along rank and then N_batch dimension
            C_buff = np.stack(C_buff, axis=-1)
            capacity_true = np.concatenate((capacity_true, C_buff), axis=0)
        
            # denoised
            C_buff = []
            for i_rank in range(1, max_sim_rank + 1):
                mean_capacity = capacity(
                    H_freq_ideal, W_batch_denoised[..., :i_rank], SNR_dB=DL_SNR_db
                ).mean(-1)
                C_buff.append(mean_capacity.detach().cpu().numpy())
            
            C_buff = np.stack(C_buff, axis=-1)
            capacity_denoised = np.concatenate((capacity_denoised, C_buff), axis=0)

            if limit_batches!=None:
                if i_batch >= limit_batches:
                    log.info(f"limit {limit_batches}")
                    break

    # ------------------------------------------------------------------
    # Расчет gap
    p = 0.5  # CDF level, used for gap estimation in (0,1)
    
    # gap true/denoised
    gap_dict = dict()
    for i in range(max_sim_rank):
    
        # Calculate CDF curves fo true
        x1 = np.sort(capacity_true[:, i])
        y1 = np.linspace(0, 1, capacity_true.shape[0])
        c1 = np.interp(p, y1, x1)
    
        #Calculate CDF curves fo true
        x2 = np.sort(capacity_denoised[:, i])
        y2 = np.linspace(0, 1, capacity_denoised.shape[0])
        c2 = np.interp(p, y2, x2)
        
        gap = 100 * (c1 - c2) / c1

        # сохраняем результат
        gap_dict[f"rank_{i+1}"] = gap

    return gap_dict


def epochs_evaluation(cfg: DictConfig):
    'Функция расчета gap для многих эпох обучения денойзера'

    # ------------------------------------------------------------------
    # Загрузка конфигурации
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed
        
        dataset_path = cfg.dataset_path
        model_dir = cfg.input_model_dir
        
        output_data_dir = cfg.output_dir
        output_fname = cfg.output_fname

        N_batch_size = cfg.N_batch_size
        limit_batches = cfg.limit_batches
        in_memory = cfg.in_memory
        num_workers = cfg.num_workers

        RW_scenario = cfg.AWGN_params.RW_scenario
        UL_SNR_db = cfg.AWGN_params.snr_db_range

    except Exception as e:
        log.error(f'Ошибка при опредлении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e

    log.info(f"Параметры успешно прочитаны")
    log.info(f"Selected Device: {device}")
    log.info(f"Start evaluation on device: {device}")
    
    # ------------------------------------------------------------------
    # Чтение датасета
    try:
        # Инициализация датасета
        dataset = my_dataset(
            h5_path=dataset_path,
            data_key="dataset",
            in_memory=in_memory,
            apply_log=False,
            apply_norm=False,
            apply_split=False
            )
    
        log.info(f"Тестовый датасет подключен")
    except Exception as e:
        log.error(f'Ошибка подключения тестовго датасета: {dataset_path}', exc_info=True)
        raise e

    if not RW_scenario:
        log.info(f"UL_SNR_db: {UL_SNR_db}")
    else:
        log.info(f"Выбран Real-world scenario. Pn_sample_dB= -75 dB")
        
    # Создаём перемешанные индексы один раз
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset)).tolist()
    shuffled_dataset = Subset(dataset, indices)
   
    loader = DataLoader(
        shuffled_dataset,
        batch_size=N_batch_size,
        drop_last=True,
        pin_memory=True if device == 'cuda' else False,
        num_workers=num_workers
    )
    
    # ------------------------------------------------------------------
    # Считываем директорию с model_states (epoch_001, epoch_002 ...) TDnCNN и сортируем их
    pathlist = sorted(glob(os.path.join(model_dir, '*.pth')))
    pathlist = [x for x in pathlist if 'epoch_' in x]
    
    if not pathlist:
        log.error("No checkpoints found!")
        return
        
    log.info(f"Found {len(pathlist)} checkpoints")

    # ------------------------------------------------------------------
    # Корзина для gap
    # gap per denoiser version (model_state)
    gap_list = []
    max_sim_rank = 4
    
    # ------------------------------------------------------------------
    # main loop per model_state
    for epoch_idx, epoch_path in enumerate(pathlist):

        # ------------------------------------------------------------------
        # Считываем checkpoint
        TDnCNN_model, epoch_n, backward_n, val_loss = read_checkpoint(epoch_path, device)
        # ------------------------------------------------------------------
        # Расчет capacity
        log.info(f"Processing checkpoint {epoch_idx + 1}: {os.path.basename(epoch_path)}")
        gap_dict = calc_gap(loader, TDnCNN_model, RW_scenario, UL_SNR_db, max_sim_rank, device, limit_batches)

        # Добавляем метки
        gap_dict['epoch'] = epoch_n
        gap_dict['backward'] = backward_n
        gap_dict['val_loss'] = val_loss
        
        # Сохраняем для текущего model_state
        gap_list.append(gap_dict) # true/denoised

    # ------------------------------------------------------------------
    # Сохранение результатов в виде словарей
    log.info("Saving evaluation as .csv")
    df = pd.DataFrame(gap_list)
    os.makedirs(cfg.output_dir, exist_ok=True)
    data_path = os.path.join(cfg.output_dir, f'{cfg.output_fname}.csv')
    df.to_csv(data_path)
    log.info("Done.")