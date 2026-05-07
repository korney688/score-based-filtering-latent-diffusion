# Train noise-consistency autoencoder on the MNIST train split.

from __future__ import annotations

import argparse
import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.autoencoder_noise_consistency import NoiseConsistencyAutoencoder


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ae_noise_consistency_mnist_full"
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
FAST_DEV_EPOCHS = 5
FAST_DEV_SUBSET_SIZE = 10_000
NOISE_SIGMA = 0.1
NOISE_LAMBDA = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def denormalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def build_dataloaders(fast_dev_run: bool) -> tuple[DataLoader, DataLoader]:
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

    if fast_dev_run:
        subset_size = min(FAST_DEV_SUBSET_SIZE, len(dataset))
        dataset = Subset(dataset, range(subset_size))

    train_size = int(len(dataset) * TRAIN_VAL_SPLIT)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader


def reconstruction_loss(model: nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reconstruction_01 = model(batch)
    reconstruction = reconstruction_01 * 2.0 - 1.0
    loss = nn.MSELoss()(reconstruction, batch)
    return loss, reconstruction


def total_loss(
    model: NoiseConsistencyAutoencoder,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss, reconstruction = reconstruction_loss(model, batch)
    noise_loss = model.noise_consistency_loss(batch, sigma=NOISE_SIGMA)
    loss = recon_loss + NOISE_LAMBDA * noise_loss
    return loss, recon_loss, noise_loss, reconstruction


def save_loss_curve(train_losses: list[float], val_losses: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Noise-Consistency Autoencoder Loss")
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
    model.eval()
    batch, _ = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        _, reconstruction = reconstruction_loss(model, batch)
    return float(nn.MSELoss()(reconstruction, batch).item())


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = FAST_DEV_EPOCHS if args.fast_dev_run else EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(fast_dev_run=args.fast_dev_run)

    model = NoiseConsistencyAutoencoder(latent_dim=LATENT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    train_losses: list[float] = []
    train_recon_losses: list[float] = []
    train_noise_losses: list[float] = []
    val_losses: list[float] = []
    val_recon_losses: list[float] = []
    val_noise_losses: list[float] = []
    best_val_loss = float("inf")
    last_example_batch = None
    last_example_reconstruction = None

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_recon_total = 0.0
        train_noise_total = 0.0
        train_items = 0

        for batch, _ in train_loader:
            batch = batch.to(device)
            loss, recon_loss, noise_loss, reconstruction = total_loss(model, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            train_total += float(loss.item()) * batch_size
            train_recon_total += float(recon_loss.item()) * batch_size
            train_noise_total += float(noise_loss.item()) * batch_size
            train_items += batch_size

            last_example_batch = batch.detach().cpu()
            last_example_reconstruction = reconstruction.detach().cpu()

        avg_train_loss = train_total / train_items
        avg_train_recon_loss = train_recon_total / train_items
        avg_train_noise_loss = train_noise_total / train_items
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_noise_losses.append(avg_train_noise_loss)

        model.eval()
        val_total = 0.0
        val_recon_total = 0.0
        val_noise_total = 0.0
        val_items = 0
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device)
                loss, recon_loss, noise_loss, _ = total_loss(model, batch)
                batch_size = batch.shape[0]
                val_total += float(loss.item()) * batch_size
                val_recon_total += float(recon_loss.item()) * batch_size
                val_noise_total += float(noise_loss.item()) * batch_size
                val_items += batch_size

        avg_val_loss = val_total / val_items
        avg_val_recon_loss = val_recon_total / val_items
        avg_val_noise_loss = val_noise_total / val_items
        val_losses.append(avg_val_loss)
        val_recon_losses.append(avg_val_recon_loss)
        val_noise_losses.append(avg_val_noise_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "train_recon_losses": train_recon_losses,
                    "train_noise_losses": train_noise_losses,
                    "val_losses": val_losses,
                    "val_recon_losses": val_recon_losses,
                    "val_noise_losses": val_noise_losses,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "latent_dim": LATENT_DIM,
                    "noise_sigma": NOISE_SIGMA,
                    "noise_lambda": NOISE_LAMBDA,
                    "fast_dev_run": args.fast_dev_run,
                },
                CHECKPOINT_PATH,
            )
            torch.save(model.encoder.state_dict(), ENCODER_PATH)

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={avg_train_loss:.6f} "
            f"train_recon={avg_train_recon_loss:.6f} "
            f"train_noise={avg_train_noise_loss:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"val_recon={avg_val_recon_loss:.6f} "
            f"val_noise={avg_val_noise_loss:.6f}"
        )

    save_loss_curve(train_losses, val_losses, LOSS_PLOT_PATH)
    if last_example_batch is not None and last_example_reconstruction is not None:
        save_reconstruction_grid(last_example_batch, last_example_reconstruction, RECON_GRID_PATH)

    reconstruction_mse = compute_reconstruction_mse(model, val_loader, device)
    training_time_sec = time.perf_counter() - start_time
    METRICS_PATH.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "latent_dim": LATENT_DIM,
                "noise_sigma": NOISE_SIGMA,
                "noise_lambda": NOISE_LAMBDA,
                "train_losses": train_losses,
                "train_recon_losses": train_recon_losses,
                "train_noise_losses": train_noise_losses,
                "val_losses": val_losses,
                "val_recon_losses": val_recon_losses,
                "val_noise_losses": val_noise_losses,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "reconstruction_mse": reconstruction_mse,
                "fast_dev_run": args.fast_dev_run,
                "training_time_sec": training_time_sec,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"checkpoint_path={CHECKPOINT_PATH}")
    print(f"encoder_path={ENCODER_PATH}")
    print(f"training_time_sec={training_time_sec:.2f}")
    print(f"final_train_loss={train_losses[-1]:.6f}")
    print(f"final_val_loss={val_losses[-1]:.6f}")
    print(f"reconstruction_mse={reconstruction_mse:.6f}")


if __name__ == "__main__":
    main()
