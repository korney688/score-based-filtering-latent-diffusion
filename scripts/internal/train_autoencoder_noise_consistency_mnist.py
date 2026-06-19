# Train noise-consistency autoencoder on a configured image dataset.

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, random_split
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.autoencoder_noise_consistency import (
    NoiseConsistencyBase,
    build_noise_consistency_autoencoder,
    normalize_noise_consistency_architecture_name,
)
from src.dataset_registry import DATASET_SPECS, build_torchvision_split, dataset_display_name, dataset_name


log = logging.getLogger(__name__)

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
TRAIN_VAL_SPLIT = 0.9
SEED = 42
DEFAULT_LATENT_DIM = 16

# Fast-dev mode keeps the same code path but uses fewer samples and epochs
FAST_DEV_EPOCHS = 5
FAST_DEV_SUBSET_SIZE = 10_000

# Extra regularization: the encoder should not change latent vectors too much
# when a small controlled image-space noise is added
NOISE_SIGMA = 0.1
NOISE_LAMBDA = 0.1


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_SPECS))
    parser.add_argument("--variant", default=None, choices=["small", "large"])
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--download", action="store_true")
    args, unknown_args = parser.parse_known_args(argv)
    args.overrides.extend(unknown_args)

    for override in args.overrides:
        key, separator, value = override.partition("=")
        if separator != "=":
            parser.error(f"Unsupported override: {override}")
        if key == "dataset":
            if value not in DATASET_SPECS:
                parser.error(f"Unsupported dataset: {value}")
            args.dataset = value
        elif key in {"variant", "architecture", "encoder.name", "encoder_name"}:
            normalized = value.strip().lower().replace("-", "_")
            if normalized in {"small", "noise_consistency_small"}:
                args.variant = "small"
            elif normalized in {"large", "noise_consistency_large"}:
                args.variant = "large"
            else:
                parser.error(f"Unsupported noise-consistency variant: {value}")
        elif key in {"latent_dim", "encoder.latent_dim", "encoder_latent_dim"}:
            args.latent_dim = int(value)
        elif key == "epochs":
            args.epochs = int(value)
        elif key in {"data_root", "local_root", "dataset.local_root", "imagenet100_root"}:
            args.data_root = value
        elif key in {"fast_dev_run", "fast_dev"}:
            args.fast_dev_run = value.lower() in {"1", "true", "yes"}
        elif key == "download":
            args.download = value.lower() in {"1", "true", "yes"}
        else:
            parser.error(f"Unsupported override: {override}")

    if args.dataset is None:
        args.dataset = "mnist"
    return args


def load_dataset_config(slug: str) -> dict:
    config_path = PROJECT_ROOT / "configs" / "dataset" / f"{slug}.yaml"
    cfg = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f"Dataset config must be a mapping: {config_path}")
    return dataset_cfg


def apply_data_root_override(dataset_cfg: dict, data_root: str | None) -> dict:
    if data_root:
        dataset_cfg["local_root"] = data_root
    return dataset_cfg


def get_encoder_config(dataset_cfg: dict) -> dict:
    encoder_cfg = dataset_cfg.get("encoder", {})
    if encoder_cfg is None:
        return {}
    if not isinstance(encoder_cfg, dict):
        raise TypeError("dataset encoder config must be a mapping")
    return encoder_cfg


def resolve_architecture_name(args: argparse.Namespace, dataset_cfg: dict) -> str:
    encoder_cfg = get_encoder_config(dataset_cfg)
    requested_name = args.variant or encoder_cfg.get("name") or "noise_consistency_small"
    return normalize_noise_consistency_architecture_name(requested_name)


def resolve_latent_dim(args: argparse.Namespace, dataset_cfg: dict) -> int:
    encoder_cfg = get_encoder_config(dataset_cfg)
    if args.latent_dim is not None:
        return int(args.latent_dim)
    return int(encoder_cfg.get("latent_dim", DEFAULT_LATENT_DIM))


def build_training_paths(dataset_cfg: dict, architecture_name: str, latent_dim: int) -> dict[str, Path]:
    slug = dataset_name(dataset_cfg)
    run_name = f"{architecture_name}_latent{latent_dim}"
    output_dir = PROJECT_ROOT / "checkpoints" / slug / "autoencoders" / run_name
    outputs_dir = PROJECT_ROOT / "outputs" / slug / "autoencoders" / run_name
    return {
        "output_dir": output_dir,
        "outputs_dir": outputs_dir,
        "checkpoint": output_dir / "autoencoder_checkpoint.pt",
        "encoder": output_dir / "E.pt",
        "loss_plot": outputs_dir / "loss_curve.png",
        "metrics": outputs_dir / "metrics.json",
        "reconstruction_grid": outputs_dir / "reconstruction_grid.png",
    }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def denormalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def build_dataloaders(
    dataset_cfg: dict,
    *,
    fast_dev_run: bool,
    download: bool,
) -> tuple[DataLoader, DataLoader]:
    dataset = build_torchvision_split(
        dataset_cfg,
        train=True,
        data_root=PROJECT_ROOT / "data",
        transform_profile="normalized",
        download=download,
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


def total_loss(
    model: NoiseConsistencyBase,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Total loss combines image reconstruction with latent noise-consistency
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
        if original.shape[1] == 1:
            axes[0, idx].imshow(original[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[1, idx].imshow(reconstructed[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[0, idx].imshow(original[idx].permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
            axes[1, idx].imshow(reconstructed[idx].permute(1, 2, 0).numpy(), vmin=0.0, vmax=1.0)
        axes[0, idx].axis("off")
        axes[1, idx].axis("off")

    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("recon")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def compute_reconstruction_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    # A quick final reconstruction check on one validation batch.
    model.eval()
    batch, _ = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        _, reconstruction = reconstruction_loss(model, batch)
    return float(nn.MSELoss()(reconstruction, batch).item())


def compute_reconstruction_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    batch, _ = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        _, reconstruction = reconstruction_loss(model, batch)
    return batch.detach().cpu(), reconstruction.detach().cpu()


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    dataset_cfg = apply_data_root_override(load_dataset_config(args.dataset), args.data_root)
    dataset_slug = dataset_name(dataset_cfg)
    dataset_label = dataset_display_name(dataset_cfg)
    architecture_name = resolve_architecture_name(args, dataset_cfg)
    latent_dim = resolve_latent_dim(args, dataset_cfg)
    paths = build_training_paths(dataset_cfg, architecture_name, latent_dim)
    start_time = time.perf_counter()

    log.info("=" * 80)
    log.info("Starting noise-consistency autoencoder training")
    log.info("=" * 80)
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Dataset: {dataset_label} ({dataset_slug})")
    log.info(f"Architecture: {architecture_name}")
    log.info(f"Checkpoint directory: {paths['output_dir']}")
    log.info(f"Output directory: {paths['outputs_dir']}")
    download = args.download or bool(dataset_cfg.get("download", False))
    log.info(
        "Training config: "
        f"epochs={args.epochs if args.epochs is not None else FAST_DEV_EPOCHS if args.fast_dev_run else EPOCHS}, "
        f"batch_size={BATCH_SIZE}, lr={LR}, train_val_split={TRAIN_VAL_SPLIT}, "
        f"seed={SEED}, latent_dim={latent_dim}, noise_sigma={NOISE_SIGMA}, "
        f"noise_lambda={NOISE_LAMBDA}, fast_dev_run={args.fast_dev_run}, download={download}"
    )

    set_seed(SEED)
    log.info(f"Random seed fixed: {SEED}")

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["outputs_dir"].mkdir(parents=True, exist_ok=True)
    log.info("Artifact directories are ready.")

    epochs = args.epochs if args.epochs is not None else FAST_DEV_EPOCHS if args.fast_dev_run else EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Selected device: {device}")

    log.info(f"Loading {dataset_label} train split and building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        dataset_cfg,
        fast_dev_run=args.fast_dev_run,
        download=download,
    )
    log.info(
        "Dataloaders are ready: "
        f"train_items={len(train_loader.dataset)}, val_items={len(val_loader.dataset)}, "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
    )

    first_batch, _ = next(iter(train_loader))
    log.info(f"First train batch shape: {tuple(first_batch.shape)}")

    model = build_noise_consistency_autoencoder(
        architecture_name,
        dataset_cfg,
        latent_dim=latent_dim,
    ).to(device)
    with torch.no_grad():
        smoke_batch = first_batch[: min(2, first_batch.shape[0])].to(device)
        smoke_latent = model.encode(smoke_batch)
        smoke_reconstruction = model(smoke_batch)
    log.info(
        "Model smoke forward: "
        f"input={tuple(smoke_batch.shape)}, latent={tuple(smoke_latent.shape)}, "
        f"reconstruction={tuple(smoke_reconstruction.shape)}"
    )
    optimizer = Adam(model.parameters(), lr=LR)
    log.info("Noise-consistency autoencoder and optimizer initialized.")

    if epochs == 0:
        training_time_sec = time.perf_counter() - start_time
        summary = {
            "dataset": dataset_slug,
            "architecture": architecture_name,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "latent_dim": latent_dim,
            "input_shape": list(smoke_batch.shape),
            "latent_shape": list(smoke_latent.shape),
            "reconstruction_shape": list(smoke_reconstruction.shape),
            "fast_dev_run": args.fast_dev_run,
            "download": download,
            "training_time_sec": training_time_sec,
            "smoke_only": True,
        }
        paths["metrics"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"Epochs=0 smoke completed; saved metrics: {paths['metrics']}")
        print(f"checkpoint_path={paths['checkpoint']}")
        print(f"encoder_path={paths['encoder']}")
        print(f"input_shape={tuple(smoke_batch.shape)}")
        print(f"latent_shape={tuple(smoke_latent.shape)}")
        print(f"reconstruction_shape={tuple(smoke_reconstruction.shape)}")
        print(f"training_time_sec={training_time_sec:.2f}")
        return

    # Keep per-epoch histories for plots, checkpoints, and metrics.json
    train_losses: list[float] = []
    train_recon_losses: list[float] = []
    train_noise_losses: list[float] = []
    val_losses: list[float] = []
    val_recon_losses: list[float] = []
    val_noise_losses: list[float] = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        log.info("-" * 80)
        log.info(f"Epoch {epoch + 1}/{epochs} started")

        model.train()
        train_total = 0.0
        train_recon_total = 0.0
        train_noise_total = 0.0
        train_items = 0
        total_train_batches = len(train_loader)
        train_log_threshold = 20

        for batch_idx, (batch, _) in enumerate(train_loader):
            # Labels are ignored: the autoencoder learns image reconstruction
            batch = batch.to(device)
            loss, recon_loss, noise_loss, reconstruction = total_loss(model, batch)

            # Standard PyTorch optimization step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch.shape[0]
            # Weight by batch size so the epoch average is exact for the dataset.
            train_total += float(loss.item()) * batch_size
            train_recon_total += float(recon_loss.item()) * batch_size
            train_noise_total += float(noise_loss.item()) * batch_size
            train_items += batch_size

            progress_percent = (batch_idx + 1) / total_train_batches * 100
            if progress_percent >= train_log_threshold:
                log.info(
                    f"Training progress: {train_log_threshold}% "
                    f"({batch_idx + 1}/{total_train_batches} batches)"
                )
                train_log_threshold += 20

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
        log.info("Validation started.")
        with torch.no_grad():
            for batch, _ in val_loader:
                # Validation uses the same loss, but without gradient updates.
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

        # Save only the best model according to validation total loss.
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
                    "architecture": architecture_name,
                    "latent_dim": latent_dim,
                    "noise_sigma": NOISE_SIGMA,
                    "noise_lambda": NOISE_LAMBDA,
                    "fast_dev_run": args.fast_dev_run,
                },
                paths["checkpoint"],
            )
            # Save the encoder separately because later pipeline stages use E(x).
            torch.save(model.encoder.state_dict(), paths["encoder"])
            log.info(f"Saved new best checkpoint: {paths['checkpoint']}")
            log.info(f"Saved encoder state: {paths['encoder']}")

        epoch_time = time.perf_counter() - epoch_start_time
        log.info(
            f"Epoch {epoch + 1}/{epochs} completed | "
            f"train_loss={avg_train_loss:.6f} | "
            f"train_recon={avg_train_recon_loss:.6f} | "
            f"train_noise={avg_train_noise_loss:.6f} | "
            f"val_loss={avg_val_loss:.6f} | "
            f"val_recon={avg_val_recon_loss:.6f} | "
            f"val_noise={avg_val_noise_loss:.6f} | "
            f"best_val_loss={best_val_loss:.6f} | "
            f"time_min={epoch_time / 60:.2f}"
        )

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={avg_train_loss:.6f} "
            f"train_recon={avg_train_recon_loss:.6f} "
            f"train_noise={avg_train_noise_loss:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"val_recon={avg_val_recon_loss:.6f} "
            f"val_noise={avg_val_noise_loss:.6f}"
        )

    # Final artifacts are generated from the saved best checkpoint, not the last training batch.
    if paths["checkpoint"].exists():
        checkpoint = torch.load(paths["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info(f"Loaded best checkpoint for final artifacts: {paths['checkpoint']}")

    # Save human-readable artifacts after training finishes.
    save_loss_curve(train_losses, val_losses, paths["loss_plot"])
    log.info(f"Saved loss curve: {paths['loss_plot']}")
    example_batch, example_reconstruction = compute_reconstruction_examples(model, val_loader, device)
    save_reconstruction_grid(example_batch, example_reconstruction, paths["reconstruction_grid"])
    log.info(f"Saved reconstruction grid from best checkpoint: {paths['reconstruction_grid']}")

    reconstruction_mse = compute_reconstruction_mse(model, val_loader, device)
    training_time_sec = time.perf_counter() - start_time
    # Save machine-readable training summary for later comparison.
    paths["metrics"].write_text(
        json.dumps(
            {
                "dataset": dataset_slug,
                "architecture": architecture_name,
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "latent_dim": latent_dim,
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
    log.info(f"Saved metrics: {paths['metrics']}")
    log.info("=" * 80)
    log.info(
        "Noise-consistency autoencoder training completed | "
        f"final_train_loss={train_losses[-1]:.6f} | "
        f"final_val_loss={val_losses[-1]:.6f} | "
        f"best_val_loss={best_val_loss:.6f} | "
        f"reconstruction_mse={reconstruction_mse:.6f} | "
        f"total_time_min={training_time_sec / 60:.2f}"
    )
    log.info("=" * 80)

    print(f"checkpoint_path={paths['checkpoint']}")
    print(f"encoder_path={paths['encoder']}")
    print(f"training_time_sec={training_time_sec:.2f}")
    print(f"final_train_loss={train_losses[-1]:.6f}")
    print(f"final_val_loss={val_losses[-1]:.6f}")
    print(f"reconstruction_mse={reconstruction_mse:.6f}")


if __name__ == "__main__":
    main()
