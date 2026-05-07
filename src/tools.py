import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class Simple_EarlyStop:
    def __init__(self, patience=20, min_delta=0, verbose=True):
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
                log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0


def set_seed(seed: int = None, device: str = "cpu"):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.cuda.manual_seed_all(seed)
