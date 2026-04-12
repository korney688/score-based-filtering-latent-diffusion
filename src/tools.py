import torch
import numpy as np

import logging

log = logging.getLogger(__name__)

class Simple_EarlyStop:
    def __init__(self, patience=20, min_delta=0, verbose=True):
        """
        Args:
            patience (int): Сколько эпох ждать после последнего улучшения Loss.
            min_delta (float): Минимальное изменение, которое считается улучшением.
            verbose (bool): Выводить ли сообщения в лог.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                log.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0


def set_seed(seed: int = None, device: str = 'cpu'):
    """Фиксация random seed"""
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed_all(seed)


def get_snr_batch(SNR_dB, size):
    """Функция формирования SNR под батч"""
    
    if np.isscalar(SNR_dB):
        return np.full(size, SNR_dB)
        
    low, high = sorted(SNR_dB)
    
    if low == high:
        high = high + 1
    
    return np.random.randint(low=low, high=high, size=size)