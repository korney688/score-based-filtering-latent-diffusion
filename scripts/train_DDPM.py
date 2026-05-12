import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from tqdm import tqdm
import logging

import sys
import os

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd()) 

from src.datasets import my_dataset
from src.DDPM_model import build_DDPM_model
from src.tools import Simple_EarlyStop, set_seed


log = logging.getLogger(__name__)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def configure_run_file_logging(output_dir: str, log_filename: str = "main.log") -> None:
    # Add a file logger for this training run
    log_path = os.path.abspath(os.path.join(output_dir, log_filename))
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == log_path:
            return

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root_logger.addHandler(file_handler)


def build_mnist_dataset():
    # Build the MNIST train split with the same normalization used by the model
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return datasets.MNIST(
        root=os.path.join(PROJECT_ROOT, "data"),
        train=True,
        download=False,
        transform=transform,
    )


def unpack_image_batch(batch):
    # DataLoader batches can be either images or (images, labels)
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def save_loss_plot(train_losses, val_losses, output_path: str) -> None:
    # Save train and validation loss curves after training
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


def get_scheduler(n_epochs, optimizer, lr_start, warmup_params, plateau_params, scheduler_mode='warmup'):
    # Choose one learning-rate schedule from the config

    if scheduler_mode == 'warmup':
        
        warmup_epochs = max(1, int(n_epochs * warmup_params['threshold']))
        scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
        return scheduler

    if scheduler_mode == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr_start/10)
        return scheduler

    if scheduler_mode == 'plateau':
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

    if scheduler_mode == 'comb':
        warmup_epochs = max(1, int(n_epochs * warmup_params['threshold']))
        scheduler1 = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs, eta_min=lr_start/10)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
        return scheduler


def train_one_epoch(model, loader: DataLoader, optimizer, device, grad_clip_val) -> float:
    # Train only the DDPM noise predictor for one epoch
    model.model.train()
    total_loss = 0.0

    total_batches = len(loader)
    log_threshold = 20
    
    for i_batch, batch in enumerate(loader):
        x_batch = unpack_image_batch(batch)
        x_batch = x_batch.to(device)
        loss = model.train_step(x_batch)

        # Backpropagate the noise prediction loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=grad_clip_val)
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_percent = (i_batch + 1) / total_batches * 100
        if progress_percent >= log_threshold:
            log.info(f"Training progress: {int(log_threshold)}% ({i_batch + 1}/{total_batches} batches)")
            log_threshold += 20

    avg_loss = total_loss / len(loader)
        
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, loader: DataLoader, device: str, epoch_desc: str) -> float:
    # Measure DDPM loss on the validation split without gradient updates
    model.model.eval()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=epoch_desc, leave=False)
    for batch in pbar:
        x_batch = unpack_image_batch(batch)
        x_batch = x_batch.to(device)
        loss = model.train_step(x_batch)
        current_loss = loss.item()
        total_loss += current_loss
        pbar.set_postfix({'val_loss': f'{current_loss:.4f}'})

    avg_loss  = total_loss / len(loader)
        
    return avg_loss


def run_train_DDPM(cfg: DictConfig):

    try:
        # Read all training parameters from the Hydra config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        seed = cfg.seed

        dataset_source = cfg.get('dataset_source', 'h5')
        h5_dataset = cfg.get('h5_dataset', {})
        dataset_path = h5_dataset.get('path', cfg.get('dataset_path', None))
        output_dir = cfg.output_dir

        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        in_memory = cfg.get('in_memory', 'False')

        lr = cfg.lr
        scheduler_mode = cfg.get('scheduler_mode', cfg.get('sheduler_mode', None))
        DDPM_params = cfg.DDPM_params
        latent_noise_mode = cfg.get('latent_noise_mode', 'baseline')
        autoencoder_kind = cfg.get('autoencoder_kind', 'baseline')
        autoencoder_checkpoint_path = cfg.get('autoencoder_checkpoint_path', './models/autoencoder.pth')
        plateau_params = cfg.get('plateau_params', None)
        warmup_params = cfg.get('warmup_params', None)
        n_epochs = cfg.n_epochs
        patience = cfg.patience
        grad_clip_val = cfg.grad_clip_val

        save_interval = cfg.get('save_interval', 'last_only')

        log.info("Training parameters were read successfully")
    except Exception as e:
        log.error(f"Failed to read training parameters. Execution stopped: {e}", exc_info=True)
        raise e

    log.info("Starting latent-DDPM training")

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_dir) and any(os.scandir(output_dir)):
        msg = f"Output directory is not empty: {output_dir}"
        log.error(msg)
        raise FileExistsError(msg)
    configure_run_file_logging(output_dir)
    log.info(f"Run log path: {os.path.join(output_dir, 'main.log')}")

    set_seed(seed, device)

    # Load the dataset either from torchvision MNIST or from the project H5 dataset
    if dataset_source == 'mnist':
        log.info("Using torchvision MNIST train split")
        full_dataset = build_mnist_dataset()
        log.info(f"MNIST dataset size: {len(full_dataset)}")
    elif dataset_source == 'h5':
        if dataset_path is None:
            raise ValueError("dataset_path or h5_dataset.path is required when dataset_source='h5'")
        if not os.path.exists(dataset_path):
            log.error(f"Missing clean data at {dataset_path}")
            raise FileNotFoundError(f"Missing clean data at {dataset_path}")

        print("DATASET PATH:", dataset_path)
        full_dataset = my_dataset(
            h5_path=dataset_path,
            data_key="dataset",
            in_memory=in_memory,
            apply_log=h5_dataset.get('apply_log', DDPM_params.get('apply_log', True)),
            apply_norm=h5_dataset.get('apply_norm', DDPM_params.get('apply_norm', True)),
            apply_split=h5_dataset.get('apply_split', DDPM_params.get('apply_split', True)),
            data_mode="image"
        )
        log.info(f'Dataset shape: {full_dataset.shape}')
    else:
        raise ValueError(f"Unsupported dataset_source: {dataset_source}")


    # Use 90% of the data for training and 10% for validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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

    log.info("Dataset is ready")

    # Log one batch to check shape and basic statistics before training
    log.info("Batch test:")
    data_batch = unpack_image_batch(next(iter(train_loader)))
    log.info(f"data Shape, dtype: {data_batch.shape}, {data_batch.dtype}")

    amp_channel = data_batch[:, 0, ...]
    if data_batch.shape[1] > 1:
        phase_channel = data_batch[:, 1, ...]
    else:
        phase_channel = None
    if phase_channel is not None:
        log.info(f"Phase     | Mean: {phase_channel.mean():.4f} | Std: {phase_channel.std():.4f}")
    log.info(f"Amplitude | Mean: {amp_channel.mean():.4f}   | Std: {amp_channel.std():.4f}")

    base_dim = DDPM_params.get('base_dim', 16)
    deep = DDPM_params.get('deep', 3)

    DDPM_params['base_dim'] = base_dim
    DDPM_params['deep'] = deep

    log.info(f"Selected deep: {deep}, base_dim: {base_dim}")
    log.info(f"Selected latent noise mode: {latent_noise_mode}")
    log.info(f"Selected autoencoder kind: {autoencoder_kind}")
    log.info(f"Selected autoencoder checkpoint: {autoencoder_checkpoint_path}")
    # Build the latent DDPM: frozen autoencoder + trainable UNet noise predictor
    DDPM_model = build_DDPM_model(
        base_dim=base_dim,
        deep=deep,
        device=device,
        latent_noise_mode=latent_noise_mode,
        autoencoder_kind=autoencoder_kind,
        autoencoder_checkpoint_path=to_absolute_path(autoencoder_checkpoint_path),
    )
    optimizer = torch.optim.Adam(DDPM_model.model.parameters(), lr=lr)
    early_stopping = Simple_EarlyStop(patience=patience, verbose=True)

    if scheduler_mode in ['warmup', 'cos', 'comb', 'plateau']:
        log.info(f"Selected scheduler: {scheduler_mode}")
        scheduler = get_scheduler(n_epochs, optimizer, lr, warmup_params, plateau_params, scheduler_mode)
    else:
        log.info(f"Scheduler OFF")


    # Store metrics and checkpoint paths during training
    target_epoch = 0 + n_epochs
    best_loss = float('inf')
    history = []
    last_checkpoint_path = None
    train_losses = []
    val_losses = []

    # Run the main epoch loop
    log.info(f"Start training on {device} for {n_epochs} epochs...")
    for epoch in range(0, target_epoch):
        log.info(f"Epoch {epoch+1}, start:")
        current_lr = optimizer.param_groups[0]['lr']

        avg_train_loss = train_one_epoch(DDPM_model, train_loader, optimizer, device, grad_clip_val)
        avg_val_loss = validate_one_epoch(DDPM_model, val_loader, device, f"Epoch {epoch+1}")

        # Update the scheduler after validation
        if scheduler_mode in ['warmup', 'cos', 'comb']:
            scheduler.step()
        elif scheduler_mode == 'plateau':
            scheduler.step(avg_val_loss)

        msg = (f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} | "
               f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        log.info(msg)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        history.append(stats)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(output_dir, 'DDPM_metrics.csv'), index=False)

        # Save enough state to continue or evaluate this DDPM later
        checkpoint = {
            'epoch': epoch + 1,
            'DDPM_params': dict(DDPM_params),
            'latent_noise_mode': latent_noise_mode,
            'autoencoder_kind': autoencoder_kind,
            'autoencoder_checkpoint_path': to_absolute_path(autoencoder_checkpoint_path),
            'model_state_dict': DDPM_model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }

        current_epoch = epoch + 1
        current_checkpoint_path = os.path.join(output_dir, f"epoch_{current_epoch:03d}.pth")

        # Either keep only the latest checkpoint or save by interval
        if save_interval == 'last_only':
            torch.save(checkpoint, current_checkpoint_path)
            log.info(f"Saved checkpoint: {os.path.basename(current_checkpoint_path)}")
            if last_checkpoint_path is not None and last_checkpoint_path != current_checkpoint_path:
                if os.path.exists(last_checkpoint_path):
                    try:
                        os.remove(last_checkpoint_path)
                        log.debug(f"Removed previous checkpoint: {os.path.basename(last_checkpoint_path)}")
                    except OSError as e:
                        log.warning(f"Failed to remove previous checkpoint: {e}")

            last_checkpoint_path = current_checkpoint_path

        else:

            try:
                interval = int(save_interval)
            except ValueError:
                interval = 5
                log.info("Invalid save_interval value. Falling back to save_interval=5")

            if current_epoch % interval == 0 or current_epoch == target_epoch:
                torch.save(checkpoint, current_checkpoint_path)
                log.info(f"Saved checkpoint: {os.path.basename(current_checkpoint_path)}")


        # Keep the best model by train loss
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            log.info(f"--> Saved New Best Model! Loss improved to {best_loss:.6f}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            log.info("Early stopping triggered. Training stopped.")
            break

    # Save a compact loss plot for quick inspection
    loss_plot_path = os.path.join(output_dir, 'ddpm_loss.png')
    save_loss_plot(train_losses, val_losses, loss_plot_path)
    log.info(f"Loss plot saved to: {loss_plot_path}")


