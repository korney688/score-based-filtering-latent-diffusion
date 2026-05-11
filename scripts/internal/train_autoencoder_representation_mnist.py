# Train representation-based autoencoder on the MNIST train split

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.autoencoder_representation import RepresentationAutoencoder


log = logging.getLogger(__name__)

# All artifacts from this representation-autoencoder run are written here
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ae_representation_mnist"
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

# Fast-dev mode keeps the same training code path with fewer samples and epochs
FAST_DEV_EPOCHS = 5
FAST_DEV_SUBSET_SIZE = 10_000

# Start with the representation backbone frozen, then optionally fine-tune it
FREEZE_EPOCHS = 5


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true")
    # Allows comparing pretrained ResNet features against randomly initialized features
    parser.add_argument("--no_pretrained", action="store_true")
    # Number of initial epochs during which only the projector and decoder are trained
    parser.add_argument("--freeze_epochs", type=int, default=FREEZE_EPOCHS)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def denormalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def build_dataloaders(fast_dev_run: bool) -> tuple[DataLoader, DataLoader]:
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

    # Optional short run for checking that the full training pipeline works
    if fast_dev_run:
        subset_size = min(FAST_DEV_SUBSET_SIZE, len(dataset))
        dataset = Subset(dataset, range(subset_size))

    # Use a fixed generator so train/validation split is reproducible
    train_size = int(len(dataset) * TRAIN_VAL_SPLIT)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def reconstruction_loss(model: nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reconstruction_01 = model(batch)
    # Decoder output is [0, 1], while training images are normalized to [-1, 1]
    reconstruction = reconstruction_01 * 2.0 - 1.0
    loss = nn.MSELoss()(reconstruction, batch)
    return loss, reconstruction


def make_optimizer(model: nn.Module) -> Adam:
    # Only parameters with requires_grad=True should be passed to the optimizer
    return Adam((param for param in model.parameters() if param.requires_grad), lr=LR)


def set_backbone_trainable(model: RepresentationAutoencoder, trainable: bool) -> None:
    # Toggle whether the ResNet representation backbone is frozen or fine-tuned
    for param in model.encoder.backbone.parameters():
        param.requires_grad = trainable


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Representation Autoencoder Reconstruction Loss")
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


def compute_reconstruction_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    # A quick final reconstruction check on one validation batch
    model.eval()
    batch, _ = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        _, reconstruction = reconstruction_loss(model, batch)
    return float(nn.MSELoss()(reconstruction, batch).item())


def main() -> None:
    configure_logging()
    args = parse_args()
    start_time = time.perf_counter()

    log.info("=" * 80)
    log.info("Starting representation autoencoder training")
    log.info("=" * 80)
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(
        "Training config: "
        f"epochs={FAST_DEV_EPOCHS if args.fast_dev_run else EPOCHS}, "
        f"batch_size={BATCH_SIZE}, lr={LR}, train_val_split={TRAIN_VAL_SPLIT}, "
        f"seed={SEED}, latent_dim={LATENT_DIM}, "
        f"pretrained_backbone={not args.no_pretrained}, "
        f"freeze_epochs={args.freeze_epochs}, fast_dev_run={args.fast_dev_run}"
    )

    set_seed(SEED)
    log.info(f"Random seed fixed: {SEED}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output directory is ready.")

    epochs = FAST_DEV_EPOCHS if args.fast_dev_run else EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Selected device: {device}")

    log.info("Loading MNIST train split and building dataloaders...")
    train_loader, val_loader = build_dataloaders(fast_dev_run=args.fast_dev_run)
    log.info(
        "Dataloaders are ready: "
        f"train_items={len(train_loader.dataset)}, val_items={len(val_loader.dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    model = RepresentationAutoencoder(
        latent_dim=LATENT_DIM,
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    ).to(device)
    # The first optimizer excludes frozen backbone parameters
    optimizer = make_optimizer(model)
    log.info("Representation autoencoder and optimizer initialized.")

    # Keep per-epoch histories for plots, checkpoints, and metrics.json
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    last_example_batch = None
    last_example_reconstruction = None
    unfroze_backbone = False

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        log.info("-" * 80)
        log.info(f"Epoch {epoch + 1}/{epochs} started")

        # After the warm-up period, include the representation backbone in training
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            set_backbone_trainable(model, True)
            # Rebuild the optimizer so newly trainable backbone parameters are included
            optimizer = make_optimizer(model)
            unfroze_backbone = True
            log.info("Backbone was unfrozen and optimizer was reinitialized")

        model.train()
        train_total = 0.0
        train_items = 0
        total_train_batches = len(train_loader)
        train_log_threshold = 20

        for batch_idx, (batch, _) in enumerate(train_loader):
            # Labels are ignored: the autoencoder learns image reconstruction
            batch = batch.to(device)
            loss, reconstruction = reconstruction_loss(model, batch)

            # Standard PyTorch optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            # Weight by batch size so the epoch average is exact for the dataset
            train_total += float(loss.item()) * batch_size
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
        train_losses.append(avg_train_loss)

        model.eval()
        val_total = 0.0
        val_items = 0
        log.info("Validation started.")
        with torch.no_grad():
            for batch, _ in val_loader:
                # Validation uses reconstruction loss only, without gradient updates
                batch = batch.to(device)
                loss, _ = reconstruction_loss(model, batch)
                batch_size = batch.shape[0]
                val_total += float(loss.item()) * batch_size
                val_items += batch_size

        avg_val_loss = val_total / val_items
        val_losses.append(avg_val_loss)

        # Save only the best model according to validation reconstruction loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "latent_dim": LATENT_DIM,
                    "pretrained_backbone": model.encoder.backbone_pretrained,
                    "freeze_epochs": args.freeze_epochs,
                    "unfroze_backbone": unfroze_backbone,
                    "fast_dev_run": args.fast_dev_run,
                },
                CHECKPOINT_PATH,
            )
            # Save the encoder separately because later pipeline stages use E(x)
            torch.save(model.encoder.state_dict(), ENCODER_PATH)
            log.info(f"Saved new best checkpoint: {CHECKPOINT_PATH}")
            log.info(f"Saved encoder state: {ENCODER_PATH}")

        epoch_time = time.perf_counter() - epoch_start_time
        log.info(
            f"Epoch {epoch + 1}/{epochs} completed | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"best_val_loss={best_val_loss:.6f} | "
            f"time_min={epoch_time / 60:.2f}"
        )

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={avg_train_loss:.6f} "
            f"val_loss={avg_val_loss:.6f}"
        )

    # Save artifacts after training finishes
    save_loss_curve(train_losses, val_losses, LOSS_PLOT_PATH)
    log.info(f"Saved loss curve: {LOSS_PLOT_PATH}")
    if last_example_batch is not None and last_example_reconstruction is not None:
        save_reconstruction_grid(last_example_batch, last_example_reconstruction, RECON_GRID_PATH)
        log.info(f"Saved reconstruction grid: {RECON_GRID_PATH}")

    reconstruction_mse = compute_reconstruction_mse(model, val_loader, device)
    training_time_sec = time.perf_counter() - start_time
    # Save training summary for later encoder comparison
    METRICS_PATH.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "latent_dim": LATENT_DIM,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "reconstruction_mse": reconstruction_mse,
                "pretrained_backbone": model.encoder.backbone_pretrained,
                "freeze_epochs": args.freeze_epochs,
                "unfroze_backbone": unfroze_backbone,
                "fast_dev_run": args.fast_dev_run,
                "training_time_sec": training_time_sec,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"Saved metrics: {METRICS_PATH}")
    log.info("=" * 80)
    log.info(
        "Representation autoencoder training completed | "
        f"final_train_loss={train_losses[-1]:.6f} | "
        f"final_val_loss={val_losses[-1]:.6f} | "
        f"best_val_loss={best_val_loss:.6f} | "
        f"reconstruction_mse={reconstruction_mse:.6f} | "
        f"total_time_min={training_time_sec / 60:.2f}"
    )
    log.info("=" * 80)

    print(f"checkpoint_path={CHECKPOINT_PATH}")
    print(f"encoder_path={ENCODER_PATH}")
    print(f"pretrained_backbone={model.encoder.backbone_pretrained}")
    print(f"training_time_sec={training_time_sec:.2f}")
    print(f"final_train_loss={train_losses[-1]:.6f}")
    print(f"final_val_loss={val_losses[-1]:.6f}")
    print(f"reconstruction_mse={reconstruction_mse:.6f}")


if __name__ == "__main__":
    main()
