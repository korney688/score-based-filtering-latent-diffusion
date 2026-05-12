import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.DDPM_model import build_DDPM_model
from src.filters import compute_latent_ddpm_scores, save_filtering_grids, save_noisy_filtering_grids, select_indices
from src.tools import set_seed

log = logging.getLogger(__name__)


def build_mnist_train_dataset():
    # Use the original MNIST train split; filtering saves indices, not images.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return datasets.MNIST(
        root=PROJECT_ROOT / "data",
        train=True,
        download=False,
        transform=transform,
    )


def resolve_checkpoint_path(cfg: DictConfig) -> Path:
    # The DDPM branch selects the checkpoint used for scoring.
    branch = cfg.ddpm_branch
    if branch not in {"baseline", "induced"}:
        raise ValueError(f"Unsupported ddpm_branch: {branch}")

    checkpoint_roots = cfg.checkpoint_roots
    checkpoint_root = Path(to_absolute_path(checkpoint_roots[branch]))
    checkpoint_path = checkpoint_root / cfg.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing DDPM checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_ddpm_from_checkpoint(cfg: DictConfig, checkpoint_path: Path, device: str):
    # Rebuild the same latent-DDPM wrapper and load only the trainable UNet weights.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_params = checkpoint.get("DDPM_params", {})

    base_dim = checkpoint_params.get("base_dim", cfg.model.base_dim)
    deep = checkpoint_params.get("deep", cfg.model.deep)
    latent_noise_mode = checkpoint.get("latent_noise_mode", cfg.ddpm_branch)
    autoencoder_kind = checkpoint.get("autoencoder_kind", cfg.model.autoencoder_kind)
    autoencoder_checkpoint_path = checkpoint.get(
        "autoencoder_checkpoint_path",
        to_absolute_path(cfg.model.autoencoder_checkpoint_path),
    )

    if latent_noise_mode != cfg.ddpm_branch:
        raise ValueError(
            f"Checkpoint latent_noise_mode={latent_noise_mode!r} does not match "
            f"ddpm_branch={cfg.ddpm_branch!r}"
        )

    ddpm_model = build_DDPM_model(
        base_dim=base_dim,
        deep=deep,
        device=device,
        latent_noise_mode=latent_noise_mode,
        autoencoder_kind=autoencoder_kind,
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
    )
    ddpm_model.model.load_state_dict(checkpoint["model_state_dict"])
    ddpm_model.eval()
    return ddpm_model


def build_output_dir(cfg: DictConfig) -> Path:
    # Keep every Stage 3 run under one experiment root.
    output_root = Path(to_absolute_path(cfg.output_root))
    if cfg.filter_mode == "top_k":
        run_name = f"top_k_{cfg.keep_ratio:g}"
    elif cfg.filter_mode == "quantile":
        run_name = f"quantile_{cfg.quantile_low:g}_{cfg.quantile_high:g}"
    else:
        raise ValueError(f"Unsupported filter_mode: {cfg.filter_mode}")
    return output_root / cfg.ddpm_branch / run_name


def write_outputs(
    output_dir: Path,
    cfg: DictConfig,
    dataset,
    checkpoint_path: Path,
    score_table,
    selected_indices: np.ndarray,
    visual_samples,
) -> None:
    # Save all artifacts needed to reproduce and reuse the selection.
    output_dir.mkdir(parents=True, exist_ok=True)

    score_table = score_table.copy()
    score_table["selected"] = score_table["dataset_index"].isin(selected_indices)

    score_table.to_csv(output_dir / "scores.csv", index=False)
    np.save(output_dir / "selected_indices.npy", selected_indices)
    OmegaConf.save(config=cfg, f=output_dir / "config.yaml", resolve=True)
    noisy_grid_files = save_noisy_filtering_grids(
        visual_samples=visual_samples,
        output_dir=output_dir,
    )
    clean_grid_files = save_filtering_grids(
        dataset=dataset,
        scores_df=score_table,
        selected_indices=selected_indices,
        output_dir=output_dir,
        n_images=cfg.grid_n_images,
    )

    metadata = {
        "ddpm_branch": cfg.ddpm_branch,
        "checkpoint_path": str(checkpoint_path),
        "filtering_mode": cfg.filter_mode,
        "keep_ratio": float(cfg.keep_ratio),
        "quantile_low": float(cfg.quantile_low),
        "quantile_high": float(cfg.quantile_high),
        "seed": int(cfg.seed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "score_definition": "score = ||eps_pred||^2",
        "dataset": "torchvision MNIST train split",
        "num_scored_samples": int(len(score_table)),
        "num_selected_samples": int(len(selected_indices)),
        "visual_grids": noisy_grid_files + clean_grid_files,
        "main_visual_grids": noisy_grid_files,
        "noisy_grid_source": "saved during the same scoring pass",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def filter_dataset(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed, device)

    checkpoint_path = resolve_checkpoint_path(cfg)
    output_dir = build_output_dir(cfg)
    if output_dir.exists() and any(output_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    log.info("Stage 3 filtering branch: %s", cfg.ddpm_branch)
    log.info("Stage 3 filtering mode: %s", cfg.filter_mode)
    log.info("DDPM checkpoint: %s", checkpoint_path)

    dataset = build_mnist_train_dataset()
    ddpm_model = load_ddpm_from_checkpoint(cfg, checkpoint_path, device)

    score_table, visual_samples = compute_latent_ddpm_scores(
        dataset=dataset,
        ddpm_model=ddpm_model,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        visual_n_images=cfg.noisy_grid_n_images,
    )
    selected_indices = select_indices(
        score_table=score_table,
        filter_mode=cfg.filter_mode,
        keep_ratio=cfg.keep_ratio,
        quantile_low=cfg.quantile_low,
        quantile_high=cfg.quantile_high,
    )

    write_outputs(output_dir, cfg, dataset, checkpoint_path, score_table, selected_indices, visual_samples)
    log.info("Saved Stage 3 filtering outputs to %s", output_dir)
