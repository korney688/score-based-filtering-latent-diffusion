import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from tqdm import tqdm
import logging
from datetime import datetime

import sys
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd()) 

# РРјРїРѕСЂС‚ РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРёС… С„СѓРЅРєС†РёР№
from src.datasets import my_dataset
from src.DDPM_model import build_DDPM_model
from src.tools import Simple_EarlyStop, set_seed


# РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ Р›РѕРіРіРµСЂР°
log = logging.getLogger(__name__)


def save_loss_plot(train_losses, val_losses, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, val_losses, label='val loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_scheduler(n_epochs, optimizer, lr_start, warmup_params, plateau_params, sheduler_mode='warmup'):
    "РћРїСЂРµРґРµР»СЏРµС‚ СЃС‚СЂСѓРєС‚СѓСЂСѓ Scheduler"

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


def train_one_epoch(model, loader: DataLoader, optimizer, device, grad_clip_val) -> float:
    """Р¦РёРєР» РѕР±СѓС‡РµРЅРёСЏ. РћРґРЅР° СЌРїРѕС…Р°"""
    model.model.train()
    total_loss = 0.0

    total_batches = len(loader)
    log_threshold = 20
    
    for i_batch, x_batch in enumerate(loader):
        x_batch = x_batch.to(device)
        loss = model.train_step(x_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=grad_clip_val)
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_percent = (i_batch + 1) / total_batches * 100
        if progress_percent >= log_threshold:
            log.info(f"РћР±СЂР°Р±РѕС‚Р°РЅРѕ {int(log_threshold)}% РґР°РЅРЅС‹С… (Р±Р°С‚С‡ {i_batch + 1}/{total_batches})")
            log_threshold += 20

    # Р’С‹С‡РёСЃР»РµРЅРёРµ СЃСЂРµРґРЅРµРіРѕ Р»РѕСЃСЃР° Р·Р° СЌРїРѕС…Сѓ
    avg_loss = total_loss / len(loader)
        
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, loader: DataLoader, device: str, epoch_desc: str) -> float:
    """Р¦РёРєР» РІР°Р»РёРґР°С†РёРё. РћРґРЅР° СЌРїРѕС…Р°"""
    model.model.eval()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=epoch_desc, leave=False)
    for x_batch in pbar:
        x_batch = x_batch.to(device)
        loss = model.train_step(x_batch)
        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})

    # Р’С‹С‡РёСЃР»РµРЅРёРµ СЃСЂРµРґРЅРµРіРѕ Р»РѕСЃСЃР° Р·Р° СЌРїРѕС…Сѓ
    avg_loss  = total_loss / len(loader)
        
    return avg_loss


# РћСЃРЅРѕРІРЅР°СЏ С„СѓРЅРєС†РёСЏ
def run_train_DDPM(cfg: DictConfig):

    # ------------------------------------------------------------------
    # Р—Р°РіСЂСѓР·РєР° РєРѕРЅС„РёРіСѓСЂР°С†РёРё
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed

        dataset_path = cfg.dataset_path
        output_dir = cfg.output_dir

        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        in_memory = cfg.get('in_memory', 'False')

        lr = cfg.lr
        sheduler_mode = cfg.sheduler_mode
        DDPM_params = cfg.DDPM_params
        plateau_params = cfg.get('plateau_params', None)
        warmup_params = cfg.get('warmup_params', None)
        n_epochs = cfg.n_epochs
        patience = cfg.patience
        grad_clip_val = cfg.grad_clip_val

        save_interval = cfg.get('save_interval', 'last_only')

        log.info("РџР°СЂР°РјРµС‚СЂС‹ СѓСЃРїРµС€РЅРѕ СЃС‡РёС‚Р°РЅС‹")
    except Exception as e:
        log.error(f'РћС€РёР±РєР° РїСЂРё РѕРїСЂРµРґРµР»РµРЅРёРё РїР°СЂР°РјРµС‚СЂРѕРІ -> РћРЎРўРђРќРћР’РљРђ. Execution: {e}', exc_info=True)
        raise e

    log.info(f"Start train_DDPM")

    # ------------------------------------------------------------------
    # РџСЂРѕРІРµСЂРєР° РїСѓС‚РµР№
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_dir) and any(os.scandir(output_dir)):
        msg = f"Output directory is not empty: {output_dir}"
        log.error(msg)
        raise FileExistsError(msg)

    # РџСЂРѕРІРµСЂРєР° РґРѕСЃС‚СѓРїРЅРѕСЃС‚Рё РІС…РѕРґРЅС‹С… РґР°РЅРЅС‹С…
    if not os.path.exists(dataset_path):
        log.error(f"Missing clean data at {dataset_path}")
        raise FileNotFoundError(f"Missing clean data at {dataset_path}")

    # ------------------------------------------------------------------
    # РќР°СЃС‚СЂРѕР№РєР° СЃРёРґРѕРІ
    set_seed(seed, device)

    # ------------------------------------------------------------------

    print("DATASET PATH:", dataset_path)
    # РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РґР°С‚Р°СЃРµС‚Р°
    full_dataset = my_dataset(
        h5_path=dataset_path,
        data_key="dataset",
        in_memory=in_memory,
        apply_log=DDPM_params['apply_log'],
        apply_norm=DDPM_params['apply_norm'],
        apply_split=DDPM_params['apply_split'],
        data_mode="image"
    )

    log.info(f'Dataset shape: {full_dataset.shape}')

    # train-val-split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РґР°С‚Р°Р»РѕР°РґРµСЂРѕРІ Рё С‚РµСЃС‚С‹
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
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    log.info("Р”Р°С‚Р°СЃРµС‚ РїРѕРґРєР»СЋС‡РµРЅ.")

    # РџСЂРѕРІРµСЂРєР° СЂР°Р·РјРµСЂРЅРѕСЃС‚Рё
    log.info("Batch test:")
    data_batch = next(iter(train_loader))
    log.info(f"data Shape, dtype: {data_batch.shape}, {data_batch.dtype}")

    # Р›РѕРіРёСЂРѕРІР°РЅРёРµ РЅРѕСЂРјР°Р»РёР·Р°С†РёРё РїРѕ Р°РјРїР»РёС‚СѓРґРµ Рё С„Р°Р·Рµ
    amp_channel = data_batch[:, 0, ...]
    if data_batch.shape[1] > 1:
        phase_channel = data_batch[:, 1, ...]
    else:
        phase_channel = None
    if phase_channel is not None:
        log.info(f"Phase     | Mean: {phase_channel.mean():.4f} | Std: {phase_channel.std():.4f}")
    log.info(f"Amplitude | Mean: {amp_channel.mean():.4f}   | Std: {amp_channel.std():.4f}")

    # ------------------------------------------------------------------
    # РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РјРѕРґРµР»Рё
    base_dim = DDPM_params.get('base_dim', 16)
    deep = DDPM_params.get('deep', 3)

    # РЎРѕС…СЂР°РЅСЏРµРј РЅР° СЃР»СѓС‡Р°Р№ РѕС‚СЃСѓС‚СЃС‚РІРёСЏ РїР°СЂР°РјРµС‚СЂРѕРІ
    DDPM_params['base_dim'] = base_dim
    DDPM_params['deep'] = deep

    log.info(f"Selected deep: {deep}, base_dim: {base_dim}")
    DDPM_model = build_DDPM_model(base_dim, deep, device)
    optimizer = torch.optim.Adam(DDPM_model.model.parameters(), lr=lr)
    early_stopping = Simple_EarlyStop(patience=patience, verbose=True)

    if sheduler_mode in ['warmup', 'cos', 'comb', 'plateau']:
        log.info(f"Selected scheduler: {sheduler_mode}")
        scheduler = get_scheduler(n_epochs, optimizer, lr, warmup_params, plateau_params, sheduler_mode)
    else:
        log.info(f"Scheduler OFF")

        # ------------------------------------------------------------------
    # РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РїРµСЂРµРјРµРЅРЅРѕР№ РґР»СЏ РѕС‚СЃР»РµР¶РёРІР°РЅРёСЏ Р»СѓС‡С€РµРіРѕ Р»РѕСЃСЃР°
    target_epoch = 0 + n_epochs
    best_loss = float('inf') # РЅР°С‡Р°Р»СЊРЅРѕРµ Р·РЅР°С‡РµРЅРёРµ
    history = [] # РґР»СЏ СЃР±РѕСЂР° СЃС‚Р°С‚РёСЃС‚РёРєРё
    last_checkpoint_path = None # РґР»СЏ СЃРѕС…СЂР°РЅРµРЅРёСЏ last_only
    train_losses = []
    val_losses = []

    # ------------------------------------------------------------------
    # main loop
    log.info(f"Start training on {device} for {n_epochs} epochs...")
    for epoch in range(0, target_epoch):
        log.info(f"Epoch {epoch+1}, start:")
        # РўРµРєСѓС‰РёР№ lr
        current_lr = optimizer.param_groups[0]['lr']
        # Train
        avg_train_loss = train_one_epoch(DDPM_model, train_loader, optimizer, device, grad_clip_val)
        # Validate
        avg_val_loss = validate_one_epoch(DDPM_model, val_loader, device, f"Epoch {epoch+1}")

        if sheduler_mode in ['warmup', 'cos', 'comb']:
            scheduler.step()
        elif sheduler_mode == 'plateau':
            scheduler.step(avg_val_loss)

        # Р›РѕРі
        msg = (f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | "
               f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        log.info(msg)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # РЎР±РѕСЂ СЃС‚Р°С‚РёСЃС‚РёРєРё, СЃРѕС…СЂР°РЅСЏРµРј РІ CSV
        stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        history.append(stats)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(output_dir, 'DDPM_metrics.csv'), index=False)

        # РЎРѕС…СЂР°РЅРµРЅРёРµ СЃРѕСЃС‚РѕСЏРЅРёСЏ
        checkpoint = {
            'epoch': epoch + 1,
            'DDPM_params': dict(DDPM_params),
            'model_state_dict': DDPM_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }

        current_epoch = epoch + 1
        current_checkpoint_path = os.path.join(output_dir, f"epoch_{current_epoch:03d}.pth")

        if save_interval == 'last_only':
            # РЎРѕС…СЂР°РЅРµРЅРёРµ С‚РµРєСѓС‰РµРіРѕ С‡РµРєРїРѕРёРЅС‚Р° (РїСЂРѕРґРѕР»Р¶РµРЅРёРµ РЅСѓРјРµСЂР°С†РёРё)
            torch.save(checkpoint, current_checkpoint_path)
            log.info(f"РЎРѕС…СЂР°РЅРµРЅ С‡РµРєРїРѕРёРЅС‚: {os.path.basename(current_checkpoint_path)}")
            # РЈРґР°Р»РµРЅРёРµ РїСЂРµРґС‹РґСѓС‰РµРіРѕ С‡РµРєРїРѕРёРЅС‚Р° (РѕСЃС‚Р°РІР»СЏРµРј С‚РѕР»СЊРєРѕ РїРѕСЃР»РµРґРЅРёР№)
            if last_checkpoint_path is not None and last_checkpoint_path != current_checkpoint_path:
                if os.path.exists(last_checkpoint_path):
                    try:
                        os.remove(last_checkpoint_path)
                        log.debug(f"РЈРґР°Р»РµРЅ СЃС‚Р°СЂС‹Р№ С‡РµРєРїРѕРёРЅС‚: {os.path.basename(last_checkpoint_path)}")
                    except OSError as e:
                        log.warning(f"РќРµ СѓРґР°Р»РѕСЃСЊ СѓРґР°Р»РёС‚СЊ СЃС‚Р°СЂС‹Р№ С‡РµРєРїРѕРёРЅС‚: {e}")

            last_checkpoint_path = current_checkpoint_path

        else:

            try:
                interval = int(save_interval)
            except ValueError:
                interval = 5
                log.info(f"РЈСЃС‚Р°РЅРѕРІР»РµРЅ base save_interval = 5")

            if current_epoch % interval == 0 or current_epoch == target_epoch:
                torch.save(checkpoint, current_checkpoint_path)
                log.info(f"РЎРѕС…СЂР°РЅРµРЅ С‡РµРєРїРѕРёРЅС‚: {os.path.basename(current_checkpoint_path)}")


        # РЎРѕС…СЂР°РЅРµРЅРёРµ Р»СѓС‡С€РµР№ РјРѕРґРµР»Рё
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            log.info(f"--> Saved New Best Model! Loss improved to {best_loss:.6f}")

        # РџСЂРѕРІРµСЂРєР° Early_stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            log.info("Early stopping triggered. Training stopped.")
            break

    loss_plot_path = os.path.join(output_dir, 'ddpm_loss.png')
    save_loss_plot(train_losses, val_losses, loss_plot_path)
    log.info(f"Loss plot saved to: {loss_plot_path}")


# Р¤СѓРЅРєС†РёСЏ РґРѕРѕР±СѓС‡РµРЅРёСЏ РјРѕРґРµР»Рё
