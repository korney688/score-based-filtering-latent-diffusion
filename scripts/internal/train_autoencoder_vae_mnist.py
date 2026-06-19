from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.autoencoder_vae import VariationalAutoencoder


log = logging.getLogger(__name__)

# All artifacts from this VAE training run are written here
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "mnist" / "autoencoders" / "ae_vae_mnist"
CHECKPOINT_PATH = OUTPUT_DIR / "autoencoder_checkpoint.pt"
ENCODER_PATH = OUTPUT_DIR / "E.pt"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_curve.png"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
RECON_GRID_PATH = OUTPUT_DIR / "reconstruction_grid.png"

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
TRAIN_VAL_SPLIT = 0.9
SEED = 42
LATENT_DIM = 16

# KL weight that controls how strongly the latent distribution is regularized
BETA = 1e-3


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def denormalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    # MNIST images are normalized to [-1, 1], matching the reconstruction target
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=str(PROJECT_ROOT / "data"),
        train=True,
        download=False,
        transform=transform,
    )

    # Use a fixed generator so train/validation split is reproducible
    train_size = int(len(dataset) * TRAIN_VAL_SPLIT)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def vae_loss(model: VariationalAutoencoder, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The VAE returns reconstruction plus latent distribution parameters
    reconstruction_01, mu, logvar = model(batch)

    # Decoder output is [0, 1], while training images are normalized to [-1, 1]
    reconstruction = reconstruction_01 * 2.0 - 1.0
    reconstruction_loss = nn.MSELoss()(reconstruction, batch)

    # KL term keeps the learned latent distribution close to a standard normal prior
    kl_divergence = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    # Total VAE loss balances reconstruction quality and latent-space regularization
    loss = reconstruction_loss + BETA * kl_divergence
    return loss, reconstruction_loss, kl_divergence, reconstruction


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("VAE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_reconstruction_grid(original: torch.Tensor, reconstructed: torch.Tensor, output_path: Path) -> None:
    num_items = min(8, original.shape[0])
    fig, axes = plt.subplots(2, num_items, figsize=(2 * num_items, 4))
    original = denormalize_to_unit_interval(original[:num_items]).cpu()
    reconstructed = denormalize_to_unit_interval(reconstructed[:num_items]).cpu()

    for idx in range(num_items):
        axes[0, idx].imshow(original[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, idx].axis("off")
        axes[1, idx].imshow(reconstructed[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("recon")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    configure_logging()
    start_time = time.time()

    log.info("=" * 80)
    log.info("Starting variational autoencoder training")
    log.info("=" * 80)
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(
        "Training config: "
        f"epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, "
        f"train_val_split={TRAIN_VAL_SPLIT}, seed={SEED}, "
        f"latent_dim={LATENT_DIM}, beta={BETA}"
    )

    set_seed(SEED)
    log.info(f"Random seed fixed: {SEED}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output directory is ready.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Selected device: {device}")

    log.info("Loading MNIST train split and building dataloaders...")
    train_loader, val_loader = build_dataloaders()
    log.info(
        "Dataloaders are ready: "
        f"train_items={len(train_loader.dataset)}, val_items={len(val_loader.dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    model = VariationalAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    log.info("Variational autoencoder and optimizer initialized.")

    # Keep per-epoch histories for plots, checkpoints, and metrics.json
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_reconstruction_losses: list[float] = []
    val_reconstruction_losses: list[float] = []
    train_kl_losses: list[float] = []
    val_kl_losses: list[float] = []
    best_val_loss = float("inf")
    last_example_batch = None
    last_example_reconstruction = None

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        log.info("-" * 80)
        log.info(f"Epoch {epoch + 1}/{EPOCHS} started")

        model.train()
        train_total = 0.0
        train_recon_total = 0.0
        train_kl_total = 0.0
        train_items = 0
        total_train_batches = len(train_loader)
        train_log_threshold = 20

        for batch_idx, (batch, _) in enumerate(train_loader):
            # Labels are ignored: the VAE learns image reconstruction and latent regularization
            batch = batch.to(device)
            loss, reconstruction_loss, kl_divergence, reconstruction = vae_loss(model, batch)

            # Standard PyTorch optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            # Weight by batch size so the epoch averages are exact for the dataset
            train_total += float(loss.item()) * batch_size
            train_recon_total += float(reconstruction_loss.item()) * batch_size
            train_kl_total += float(kl_divergence.item()) * batch_size
            train_items += batch_size

            # Store examples from the latest batch for the final reconstruction grid
            last_example_batch = batch.detach().cpu()
            last_example_reconstruction = reconstruction.detach().cpu()

            progress_percent = (batch_idx + 1) / total_train_batches * 100
            if progress_percent >= train_log_threshold:
                log.info(
                    f"Training progress: {train_log_threshold}% "
                    f"({batch_idx + 1}/{total_train_batches} batches)"
                )
                train_log_threshold += 20

        avg_train_loss = train_total / train_items
        avg_train_recon = train_recon_total / train_items
        avg_train_kl = train_kl_total / train_items
        train_losses.append(avg_train_loss)
        train_reconstruction_losses.append(avg_train_recon)
        train_kl_losses.append(avg_train_kl)

        model.eval()
        val_total = 0.0
        val_recon_total = 0.0
        val_kl_total = 0.0
        val_items = 0
        log.info("Validation started.")
        with torch.no_grad():
            for batch, _ in val_loader:
                # Validation uses the same VAE loss, but without gradient updates
                batch = batch.to(device)
                loss, reconstruction_loss, kl_divergence, _ = vae_loss(model, batch)
                batch_size = batch.shape[0]
                val_total += float(loss.item()) * batch_size
                val_recon_total += float(reconstruction_loss.item()) * batch_size
                val_kl_total += float(kl_divergence.item()) * batch_size
                val_items += batch_size

        avg_val_loss = val_total / val_items
        avg_val_recon = val_recon_total / val_items
        avg_val_kl = val_kl_total / val_items
        val_losses.append(avg_val_loss)
        val_reconstruction_losses.append(avg_val_recon)
        val_kl_losses.append(avg_val_kl)

        # Save only the best model according to validation total VAE loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_reconstruction_losses": train_reconstruction_losses,
                    "val_reconstruction_losses": val_reconstruction_losses,
                    "train_kl_losses": train_kl_losses,
                    "val_kl_losses": val_kl_losses,
                    "beta": BETA,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "latent_dim": LATENT_DIM,
                },
                CHECKPOINT_PATH,
            )
            # Save the encoder-side VAE components needed to compute latent distributions later
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "fc_mu": model.fc_mu.state_dict(),
                    "fc_logvar": model.fc_logvar.state_dict(),
                },
                ENCODER_PATH,
            )
            log.info(f"Saved new best checkpoint: {CHECKPOINT_PATH}")
            log.info(f"Saved encoder state: {ENCODER_PATH}")

        epoch_time = time.time() - epoch_start_time
        log.info(
            f"Epoch {epoch + 1}/{EPOCHS} completed | "
            f"train_loss={avg_train_loss:.6f} | "
            f"train_recon={avg_train_recon:.6f} | "
            f"train_kl={avg_train_kl:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"val_recon={avg_val_recon:.6f} | "
            f"val_kl={avg_val_kl:.6f} | "
            f"best_val_loss={best_val_loss:.6f} | "
            f"time_min={epoch_time / 60:.2f}"
        )

        print(
            f"epoch={epoch + 1}/{EPOCHS} "
            f"train_loss={avg_train_loss:.6f} "
            f"train_recon={avg_train_recon:.6f} "
            f"train_kl={avg_train_kl:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"val_recon={avg_val_recon:.6f} "
            f"val_kl={avg_val_kl:.6f}"
        )

    # Save artifacts after training finishes
    save_loss_curve(train_losses, val_losses, LOSS_PLOT_PATH)
    log.info(f"Saved loss curve: {LOSS_PLOT_PATH}")
    if last_example_batch is not None and last_example_reconstruction is not None:
        save_reconstruction_grid(last_example_batch, last_example_reconstruction, RECON_GRID_PATH)
        log.info(f"Saved reconstruction grid: {RECON_GRID_PATH}")

    # Save training summary for later encoder comparison
    METRICS_PATH.write_text(
        json.dumps(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "latent_dim": LATENT_DIM,
                "beta": BETA,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_reconstruction_losses": train_reconstruction_losses,
                "val_reconstruction_losses": val_reconstruction_losses,
                "train_kl_losses": train_kl_losses,
                "val_kl_losses": val_kl_losses,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"Saved metrics: {METRICS_PATH}")

    total_time = time.time() - start_time
    log.info("=" * 80)
    log.info(
        "Variational autoencoder training completed | "
        f"final_train_loss={train_losses[-1]:.6f} | "
        f"final_val_loss={val_losses[-1]:.6f} | "
        f"best_val_loss={best_val_loss:.6f} | "
        f"total_time_min={total_time / 60:.2f}"
    )
    log.info("=" * 80)

    print(f"checkpoint_path={CHECKPOINT_PATH}")
    print(f"encoder_path={ENCODER_PATH}")
    print(f"final_train_loss={train_losses[-1]:.6f}")
    print(f"final_val_loss={val_losses[-1]:.6f}")


if __name__ == "__main__":
    main()
