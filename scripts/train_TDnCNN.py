import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from tqdm import tqdm
import logging

import sys
import os
import shutil
import time
from datetime import datetime

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd()) 

# Импорт пользовательских функций
from src.datasets import my_dataset
from src.tools import Simple_EarlyStop, set_seed

from src.input_repo.tools import get_noise
from src.input_repo.TDnCNN import build_DnCNN_3D_model


# Инициализация Логгера
log = logging.getLogger(__name__)

def get_scheduler(n_epochs, optimizer, lr_start, warmup_params, plateau_params, sheduler_mode='warmup'):
    "Определяет структуру Scheduler"

    if sheduler_mode == 'warmup':
        
        warmup_epochs = max(1, int(n_epochs * warmup_params['threshold']))
        scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
        return scheduler

    if sheduler_mode == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr_start/10)
        return scheduler

    if sheduler_mode == 'plateau':
        if plateau_params is None:
            plateau_params=dict()
            plateau_params['patience']=5
            plateau_params['threshold']=0.001
            plateau_params['factor']=0.5
            
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',           
            factor=plateau_params['factor'],
            patience=plateau_params['patience'],
            threshold=plateau_params['threshold'],
            threshold_mode='rel',
            cooldown=0,
            min_lr=lr_start/100
        )

        return scheduler 

    if sheduler_mode == 'comb':
        warmup_epochs = max(1, int(n_epochs * warmup_params['threshold']))
        scheduler1 = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs, eta_min=lr_start/10)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
        return scheduler
        

def get_snr_batch(SNR_dB, size):
    """Функция формирования SNR под батч"""
    
    if np.isscalar(SNR_dB):
        return np.full(size, SNR_dB)
        
    low, high = sorted(SNR_dB)
    
    if low == high:
        high = high + 1
    
    return np.random.randint(low=low, high=high, size=size)


def train_one_epoch(model, 
                    loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    device: str,
                    SNR_dB,
                    grad_clip_val) -> float:
    """Цикл обучения. Одна эпоха"""
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    backward_n = 0
    
    total_batches = len(loader)
    log_threshold = 20
    for i_batch, x_batch in enumerate(loader):
        current_bs = x_batch.shape[0]
        current_snr = get_snr_batch(SNR_dB, current_bs)
        Noise_batch = get_noise(x_batch, SNR_dB=current_snr)
        x_batch, Noise_batch = x_batch.to(device), Noise_batch.to(device)

        x_noisy = x_batch + Noise_batch
        x_noisy = x_noisy.to(torch.complex64) # Приведение типов
        x_pred = model(x_noisy)

        loss = (x_pred - x_batch).abs().pow(2).sum()/current_bs
        
        optimizer.zero_grad()
        loss.backward()
        backward_n += 1
        #batch_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val)
        #total_grad_norm += batch_grad_norm.item()
        optimizer.step()
        
        total_loss += loss.item()

        progress_percent = (i_batch + 1) / total_batches * 100
        if progress_percent >= log_threshold:
            log.info(f"Обработано {int(log_threshold)}% данных (батч {i_batch + 1}/{total_batches})")
            log_threshold += 20

    #avg_epoch_grad_norm = total_grad_norm / len(loader)
    log.info(f'final SNR {current_snr}')
    #log.info(f"avg_epoch_grad_norm = {avg_epoch_grad_norm}")
    # Вычисление среднего лосса за эпоху
    avg_loss = total_loss / len(loader)
        
    return avg_loss, backward_n


@torch.no_grad()
def validate_one_epoch(model,
                       loader: DataLoader, 
                       device: str, 
                       epoch_desc: str, 
                       SNR_dB) -> float:
    """Цикл валидации. Одна эпоха"""
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=epoch_desc, leave=False)
    for x_batch in pbar:
        current_bs = x_batch.shape[0]
        current_snr = get_snr_batch(SNR_dB, current_bs)
        
        Noise_batch = get_noise(x_batch, SNR_dB=current_snr)
        x_batch, Noise_batch = x_batch.to(device), Noise_batch.to(device)

        x_noisy = x_batch + Noise_batch
        x_noisy = x_noisy.to(torch.complex64) # Приведение типов
        x_pred = model(x_noisy)

        loss = (x_pred - x_batch).abs().pow(2).sum()/current_bs
        
        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})

    # Вычисление среднего лосса за эпоху
    avg_loss  = total_loss / len(loader)
        
    return avg_loss


# Основная функция
def run_train_TDnCNN(cfg: DictConfig):
    
    # ------------------------------------------------------------------
    # Загрузка конфигурации
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed
        
        dataset_path = cfg.dataset_path
        output_dir = cfg.output_dir
        
        model_size = cfg.model_size
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        in_memory = cfg.get('in_memory', 'False')
        
        lr = cfg.lr
        sheduler_mode = cfg.sheduler_mode
        plateau_params = cfg.get('plateau_params', None)
        warmup_params = cfg.get('warmup_params', None)
        grad_clip_val = cfg.grad_clip_val
        n_epochs = cfg.n_epochs
        patience = cfg.patience
        SNR_dB = cfg.AWGN_params.snr_db_range

        save_interval = cfg.get('save_interval', 'last_only')
        
        log.info("Параметры успешно считаны")
    except Exception as e:
        log.error(f'Ошибка при определении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e
    
    log.info(f"Start train_TDnCNN")
    
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
        in_memory=in_memory,
        apply_log=False,
        apply_norm=False,
        apply_split=False
        )
    
    log.info(f'Dataset shape: {full_dataset.shape}')
    
    # train-val-split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Инициализация даталоадеров и тесты
    train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True if device == 'cuda' else False
        )

    val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True if device == 'cuda' else False
        )
    
    log.info("Датасет подключен.")
        
    # Проверка размерности
    log.info("Batch test:")
    data_batch = next(iter(train_loader))
    log.info(f"data Shape, dtype: {data_batch.shape}, {data_batch.dtype}") 

    # ------------------------------------------------------------------
    # Инициализация модели
    model = build_DnCNN_3D_model(device, model_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = Simple_EarlyStop(patience=patience, verbose=True)

    if sheduler_mode in ['warmup', 'cos', 'comb', 'plateau']:
        log.info(f"Selected scheduler: {sheduler_mode}") 
        scheduler = get_scheduler(n_epochs, optimizer, lr, warmup_params, plateau_params, sheduler_mode)
    else:
        log.info(f"Scheduler OFF") 

    # ------------------------------------------------------------------
    # Инициализация переменной для отслеживания лучшего лосса
    target_epoch = 0 + n_epochs
    best_loss = float('inf')
    history = []   # для статистики
    backward_n = 0 # счетчик backward
    last_checkpoint_path = None

    # ------------------------------------------------------------------
    # main loop  
    log.info(f"Start training on {device} for {n_epochs} epochs...")
    start_time = time.time()
    for epoch in range(0, target_epoch):
        
        log.info(f"Epoch {epoch+1}, start:")

        # Текущий lr
        current_lr = optimizer.param_groups[0]['lr']
        # Train
        avg_train_loss, epoch_backward_n = train_one_epoch(model, train_loader, optimizer, device, SNR_dB, grad_clip_val)
        backward_n = backward_n + epoch_backward_n
        # Validate
        avg_val_loss = validate_one_epoch(model, val_loader, device, f"Epoch {epoch+1}", SNR_dB)

        if sheduler_mode in ['warmup', 'cos', 'comb']:
            scheduler.step()

        elif sheduler_mode == 'plateau':
            scheduler.step(avg_train_loss)
            
        
        epoch_time = time.time() - start_time

        # Лог
        msg = (f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | "
                   f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e} | time: {epoch_time/60:.0f}")
        log.info(msg)

        # Сбор статистики, сохраняем в CSV
        stats = {
            'train_time': epoch_time,
            'epoch': epoch + 1,
            'backward_n': backward_n,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        history.append(stats)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(output_dir, 'TDnCNN_metrics.csv'), index=False)

        # Сохранение состояния
        checkpoint = {
            'epoch': epoch + 1,
            'backward_n': backward_n,
            'model_size': model_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }

        # Сохранение на диск
        current_epoch = epoch + 1
        current_checkpoint_path = os.path.join(output_dir, f"epoch_{current_epoch:03d}.pth")

        if save_interval == 'last_only':
            # Сохранение текущего чекпоинта (продолжение нумерации)
            torch.save(checkpoint, current_checkpoint_path)
            log.info(f"Сохранен чекпоинт: {os.path.basename(current_checkpoint_path)}")
            # Удаление предыдущего чекпоинта (оставляем только последний)
            if last_checkpoint_path is not None and last_checkpoint_path != current_checkpoint_path:
                if os.path.exists(last_checkpoint_path):
                    try:
                        os.remove(last_checkpoint_path)
                        log.debug(f"Удален старый чекпоинт: {os.path.basename(last_checkpoint_path)}")
                    except OSError as e:
                        log.warning(f"Не удалось удалить старый чекпоинт: {e}")
                        
            last_checkpoint_path = current_checkpoint_path

        else:

            try:
                interval = int(save_interval)
            except ValueError:
                interval = 5
                log.info(f"Установлен base save_interval = 5")
                
            if current_epoch % interval == 0 or current_epoch == target_epoch:
                torch.save(checkpoint, current_checkpoint_path)
                log.info(f"Сохранен чекпоинт: {os.path.basename(current_checkpoint_path)}")
        
        # Сохранение лучшей модели
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            log.info(f"--> Saved New Best Model! Loss improved to {best_loss:.6f}")

        # Проверка Early_stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            log.info("Early stopping triggered. Training stopped.")
            break



# Функция дообучения модели
def run_finetune_TDnCNN(cfg: DictConfig):
    """
    Функция дообучения (Finetuning) модели.
    """
    # ------------------------------------------------------------------
    # Загрузка конфигурационных параметров
    try:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed
        
        dataset_path = cfg.dataset_path
    
        checkpoint_dir = cfg.checkpoint_dir
        checkpoint_name = cfg.checkpoint_name

        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        in_memory = cfg.get('in_memory', 'False')
        
        lr = cfg.lr
        sheduler_mode = cfg.sheduler_mode
        optimizer_load=cfg.optimizer_load
        plateau_params = cfg.get('plateau_params', None)
        warmup_params = cfg.get('warmup_params', None)
        grad_clip_val = cfg.grad_clip_val
        n_epochs = cfg.n_epochs
        patience = cfg.patience
        SNR_dB = cfg.AWGN_params.snr_db_range

        save_interval = cfg.get('save_interval', 'last_only')
    
        log.info("Параметры успешно считаны")
        
    except Exception as e:
        log.error(f'Ошибка при определении параметров -> ОСТАНОВКА. Execution: {e}', exc_info=True)
        raise e

    # ------------------------------------------------------------------
    # Проверка доступа к предобученной модели
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        log.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Файл не найден: {checkpoint_path}")
    
    log.info(f"Запуск процесса дообучения. Загрузка из: {checkpoint_name}")

    # ------------------------------------------------------------------
    # Настройка сидов
    set_seed(seed, device)

    # ------------------------------------------------------------------
    # Инициализация датасета
    full_dataset = my_dataset(
        h5_path=dataset_path,
        data_key="dataset",
        in_memory=in_memory,
        apply_log=False,
        apply_norm=False,
        apply_split=False
        )
    
    log.info(f'Dataset shape: {full_dataset.shape}')
    
    # train-val-split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Инициализация даталоадеров и тесты
    train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True if device == 'cuda' else False
        )

    val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True if device == 'cuda' else False
        )
    
    log.info(f"Датасет для дообучения подключен. Device: {device}")
        
    # Проверка размерности
    log.info("Batch test:")
    data_batch = next(iter(train_loader))
    log.info(f"data Shape, dtype: {data_batch.shape}, {data_batch.dtype}") 

    # ------------------------------------------------------------------
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ------------------------------------------------------------------
    # Инициализация и загрузка состояния
    log.info("Инициализация модели и загрузка весов...")
    
    # Сначала создаем "чистые" объекты
    model_size = checkpoint['model_size']
    model = build_DnCNN_3D_model(device, model_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = Simple_EarlyStop(patience=patience, verbose=True)

    if sheduler_mode in ['warmup', 'cos', 'comb', 'plateau']:
        log.info(f"Selected scheduler: {sheduler_mode}") 
        scheduler = get_scheduler(n_epochs, optimizer, lr, warmup_params, plateau_params, sheduler_mode)
    else:
        log.info(f"Scheduler OFF") 
    
    # Восстанавливаем веса
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer_load == True:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Определяем начальное состояние
    start_epoch = checkpoint['epoch'] 
    backward_n = checkpoint['backward_n']
    best_loss = checkpoint.get('val_loss', float('inf'))
    
    log.info(f"Успешно загружено состояние. Старт с эпохи: {start_epoch + 1}")
    log.info(f"Предыдущий лучший Val Loss: {best_loss:.6f}")

    # ------------------------------------------------------------------
    # Инициализация цикла обучения
    history = []
    last_checkpoint_path = None # оставляем Nane, чтобы не удалять предыдущий
    
    # Формируем имя файла метрик с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"TDnCNN_metrics_finetune_{timestamp}.csv"
    metrics_path = os.path.join(checkpoint_dir, metrics_filename)
    log.info(f"Метрики дообучения будут сохранены в: {metrics_filename}")

    # Диапазон: от start_epoch до start_epoch + n_epochs
    target_epoch = start_epoch + n_epochs
    log.info(f"Start finetuning for {n_epochs} additional epochs (Total target: {target_epoch})...")

    # ------------------------------------------------------------------
    # main loop
    for epoch in range(start_epoch, target_epoch):
        
        log.info(f"Epoch {epoch+1}, start:")

        # Текущий lr
        current_lr = optimizer.param_groups[0]['lr']
        # Train
        avg_train_loss, epoch_backward_n = train_one_epoch(model, train_loader, optimizer, device, SNR_dB, grad_clip_val)
        backward_n = backward_n + epoch_backward_n
        # Validate
        avg_val_loss = validate_one_epoch(model, val_loader, device, f"Epoch {epoch+1}", SNR_dB)

        if sheduler_mode in ['warmup', 'cos', 'comb']:
            scheduler.step()

        elif sheduler_mode == 'plateau':
            scheduler.step(avg_val_loss)

        # Лог
        msg = (f"Finetune Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | "
               f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        log.info(msg)

        # Сбор статистики, сохраняем в CSV
        stats = {
            'epoch': epoch + 1,
            'backward_n': backward_n,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        history.append(stats)
        df = pd.DataFrame(history)
        df.to_csv(metrics_path, index=False)

        # Сохранение чекпоинтов
        checkpoint = {
            'epoch': epoch + 1,
            'backward_n': backward_n,
            'model_size': model_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'config': OmegaConf.to_container(cfg, resolve=True) # Сохраняем текущий конфиг
        }
        
        # Сохранение на диск
        current_epoch = epoch + 1
        current_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{current_epoch:03d}.pth")

        if save_interval == 'last_only':
            # 1. Сохранение текущего чекпоинта (продолжение нумерации)
            torch.save(checkpoint, current_checkpoint_path)
            log.info(f"Сохранен чекпоинт: {os.path.basename(current_checkpoint_path)}")
            # Удаление предыдущего чекпоинта (оставляем только последний)
            if last_checkpoint_path is not None and last_checkpoint_path != current_checkpoint_path:
                if os.path.exists(last_checkpoint_path):
                    try:
                        os.remove(last_checkpoint_path)
                        log.debug(f"Удален старый чекпоинт: {os.path.basename(last_checkpoint_path)}")
                    except OSError as e:
                        log.warning(f"Не удалось удалить старый чекпоинт: {e}")
                        
            last_checkpoint_path = current_checkpoint_path

        else:
            try:
                interval = int(save_interval)
            except ValueError:
                interval = 5
                log.info(f"Установлен base save_interval = 5")
                
            if current_epoch % interval == 0 or current_epoch == target_epoch:
                torch.save(checkpoint, current_checkpoint_path)
                log.info(f"Сохранен чекпоинт: {os.path.basename(current_checkpoint_path)}")
        
        # 2. Сохранение лучшей модели
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            log.info(f"--> Saved New Best Model (Finetune)! Loss improved to {best_loss:.6f}")

        # Проверка Early_stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            log.info("Early stopping triggered during finetuning. Training stopped.")
            break
            
    log.info("Finetuning completed.")