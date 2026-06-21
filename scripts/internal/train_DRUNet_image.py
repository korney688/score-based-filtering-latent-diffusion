from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.error import URLError

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import make_grid, save_image

from src.DRUNet_image import build_drunet
from src.TDnCNN_image import compute_lpips, compute_metrics, compute_ssim_item, create_fid_metric, prepare_for_fid
from src.tdncnn_datasets import build_dataset_split, limit_dataset, load_filtered_indices


class OnlineNoisyDatasetWithSigma(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        sigma_min: float = 0.1,
        sigma_max: float = 0.8,
        fixed_sigma: float | None = None,
    ):
        self.dataset = dataset
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.fixed_sigma = fixed_sigma

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        x_clean = sample[0] if isinstance(sample, (tuple, list)) else sample

        if self.fixed_sigma is None:
            sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max)
        else:
            sigma = torch.full((1,), float(self.fixed_sigma))
        epsilon = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma.view(1, 1, 1) * epsilon

        return x_noisy.to(torch.float32), x_clean.to(torch.float32), sigma.to(torch.float32)


def create_drunet_dataloaders(
    filtered_indices=None,
    batch_size: int = 32,
    mode: str = "full",
    seed: int = 42,
    data_root: str | Path = "data",
    dataset_name: str = "imagenet100",
    download: bool = False,
    sigma_min: float = 0.1,
    sigma_max: float = 0.8,
    fixed_sigma: float | None = None,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
):
    mode = mode.lower()
    if mode not in {"full", "filtered"}:
        raise ValueError(f"Unsupported mode: {mode}")

    train_base = build_dataset_split(train=True, data_root=data_root, dataset_name=dataset_name, download=download)
    test_base = build_dataset_split(train=False, data_root=data_root, dataset_name=dataset_name, download=download)

    loaded_filtered_indices = load_filtered_indices(filtered_indices)
    if mode == "filtered":
        if loaded_filtered_indices is None:
            raise ValueError("filtered_indices must be provided when mode='filtered'")
        train_base = Subset(train_base, np.sort(loaded_filtered_indices).tolist())

    train_base = limit_dataset(train_base, max_train_samples)
    test_base = limit_dataset(test_base, max_test_samples)

    train_dataset = OnlineNoisyDatasetWithSigma(
        train_base,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        fixed_sigma=fixed_sigma,
    )
    test_dataset = OnlineNoisyDatasetWithSigma(
        test_base,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        fixed_sigma=fixed_sigma,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Mode: {mode}")
    print(f"Train objects: {len(train_dataset)}")
    print(f"Test objects: {len(test_dataset)}")

    return train_loader, test_loader


def create_lpips_model(device: torch.device) -> nn.Module:
    try:
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except (URLError, OSError, RuntimeError) as error:
        print("LPIPS alexnet weights could not be loaded. Falling back to random trunk weights.")
        print(f"LPIPS init error: {error}")
        model = lpips.LPIPS(net="alex", pnet_rand=True).to(device)
        model.eval()
        return model


def save_loss_curve(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics_table(history: dict[str, list[float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "psnr", "ssim", "lpips", "fid"])
        for epoch_idx in range(len(history["train_loss"])):
            writer.writerow(
                [
                    epoch_idx + 1,
                    float(history["train_loss"][epoch_idx]),
                    float(history["val_loss"][epoch_idx]),
                    float(history["psnr"][epoch_idx]),
                    float(history["ssim"][epoch_idx]),
                    float(history["lpips"][epoch_idx]),
                    float(history["fid"][epoch_idx]),
                ]
            )


def save_visualization(x_noisy: torch.Tensor, x_clean: torch.Tensor, x_pred: torch.Tensor, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["Clean", "Noisy", "DRUNet"]
    tensors = [x_clean[0], x_noisy[0], x_pred[0]]
    for ax, title, tensor in zip(axes, titles, tensors):
        image = tensor.detach().cpu().numpy()
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image.transpose(1, 2, 0).clip(0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def psnr_from_mse(mse_value: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(1.0 / torch.sqrt(mse_value + 1e-8))


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for x_noisy, x_clean, sigma in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        sigma = sigma.to(device)

        x_pred = model(x_noisy, sigma=sigma)
        loss = criterion(x_pred, x_clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_noisy.shape[0]

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    lpips_model: nn.Module,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_items = 0

    fid = create_fid_metric(device)

    sample_noisy = None
    sample_clean = None
    sample_pred = None

    for x_noisy, x_clean, sigma in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        sigma = sigma.to(device)

        x_pred = model(x_noisy, sigma=sigma)
        loss = criterion(x_pred, x_clean)

        batch_size = x_noisy.shape[0]
        total_loss += loss.item() * batch_size

        mse_per_item = ((x_pred - x_clean) ** 2).flatten(1).mean(dim=1)
        psnr_per_item = psnr_from_mse(mse_per_item)
        total_psnr += float(psnr_per_item.sum().item())

        total_lpips += compute_lpips(x_pred, x_clean, lpips_model) * batch_size

        if fid is not None:
            fid.update(prepare_for_fid(x_clean), real=True)
            fid.update(prepare_for_fid(x_pred), real=False)

        x_pred_np = x_pred.detach().cpu().numpy()
        x_clean_np = x_clean.detach().cpu().numpy()
        for idx in range(batch_size):
            total_ssim += compute_ssim_item(x_pred_np[idx], x_clean_np[idx])

        total_items += batch_size

        if sample_noisy is None:
            sample_noisy = x_noisy.detach().cpu()
            sample_clean = x_clean.detach().cpu()
            sample_pred = x_pred.detach().cpu()

    if sample_noisy is None or sample_clean is None or sample_pred is None:
        raise RuntimeError("Evaluation loader is empty.")

    metrics = {
        "val_loss": total_loss / total_items,
        "psnr": total_psnr / total_items,
        "ssim": total_ssim / total_items,
        "lpips": total_lpips / total_items,
        "fid": float(fid.compute().item()) if fid is not None else float("nan"),
    }
    if fid is not None:
        fid.reset()

    return metrics, sample_noisy, sample_clean, sample_pred


def save_per_image_metrics_and_qualitative(
    model: nn.Module,
    loader,
    device: torch.device,
    lpips_model: nn.Module,
    output_dir: Path,
    run_name: str,
    max_qualitative_samples: int = 8,
) -> None:
    rows = []
    qualitative_noisy = None
    qualitative_clean = None
    qualitative_pred = None
    sample_index = 0

    model.eval()
    with torch.no_grad():
        for x_noisy, x_clean, sigma in loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            sigma = sigma.to(device)
            x_pred = model(x_noisy, sigma=sigma).clamp(0.0, 1.0)

            denoised_mse = ((x_pred - x_clean) ** 2).flatten(start_dim=1).mean(dim=1)
            noisy_mse = ((x_noisy - x_clean) ** 2).flatten(start_dim=1).mean(dim=1)
            denoised_psnr = psnr_from_mse(denoised_mse)
            noisy_psnr = psnr_from_mse(noisy_mse)

            x_pred_np = x_pred.detach().cpu().numpy()
            x_clean_np = x_clean.detach().cpu().numpy()
            x_noisy_np = x_noisy.detach().cpu().numpy()

            for idx in range(x_noisy.shape[0]):
                pred_item = x_pred[idx : idx + 1]
                clean_item = x_clean[idx : idx + 1]
                noisy_item = x_noisy[idx : idx + 1]
                rows.append(
                    {
                        "sample_index": sample_index,
                        "run_name": run_name,
                        "mse": float(denoised_mse[idx].item()),
                        "psnr": float(denoised_psnr[idx].item()),
                        "ssim": compute_ssim_item(x_pred_np[idx], x_clean_np[idx]),
                        "lpips": compute_lpips(pred_item, clean_item, lpips_model),
                        "noisy_mse": float(noisy_mse[idx].item()),
                        "noisy_psnr": float(noisy_psnr[idx].item()),
                        "noisy_ssim": compute_ssim_item(x_noisy_np[idx], x_clean_np[idx]),
                        "noisy_lpips": compute_lpips(noisy_item, clean_item, lpips_model),
                    }
                )
                sample_index += 1

            if qualitative_noisy is None:
                qualitative_noisy = x_noisy[:max_qualitative_samples].detach().cpu().clamp(0.0, 1.0)
                qualitative_clean = x_clean[:max_qualitative_samples].detach().cpu().clamp(0.0, 1.0)
                qualitative_pred = x_pred[:max_qualitative_samples].detach().cpu().clamp(0.0, 1.0)

    if rows:
        metrics_path = output_dir / "per_image_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if qualitative_noisy is not None and qualitative_clean is not None and qualitative_pred is not None:
        combined = []
        for idx in range(qualitative_noisy.shape[0]):
            combined.extend([qualitative_clean[idx], qualitative_noisy[idx], qualitative_pred[idx]])
        save_image(make_grid(torch.stack(combined, dim=0), nrow=3, padding=2), output_dir / "qualitative_triplets.png")


def run_experiment(
    experiment_name: str,
    mode: str,
    filtered_indices_path: Path | None,
    output_dir: Path,
    data_root: Path,
    checkpoint_dir: Path | None = None,
    dataset_name: str = "imagenet100",
    download: bool = False,
    in_channels: int = 3,
    batch_size: int = 8,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    scheduler: str | None = None,
    seed: int = 42,
    sigma_mode: str = "uniform",
    fixed_sigma: float | None = None,
    sigma_min: float = 0.1,
    sigma_max: float = 0.8,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    qualitative_dir = output_dir / "qualitative"
    results_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sigma_mode = sigma_mode.lower()
    if sigma_mode not in {"uniform", "fixed"}:
        raise ValueError(f"Unsupported sigma_mode: {sigma_mode}")
    if sigma_mode == "fixed" and fixed_sigma is None:
        raise ValueError("fixed_sigma must be provided when sigma_mode='fixed'")
    if sigma_mode == "uniform":
        fixed_sigma = None

    train_loader, test_loader = create_drunet_dataloaders(
        filtered_indices=filtered_indices_path,
        batch_size=batch_size,
        mode=mode,
        seed=seed,
        data_root=data_root,
        dataset_name=dataset_name,
        download=download,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        fixed_sigma=fixed_sigma,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )

    model_config = dict(model_config or {})
    model = build_drunet(
        in_channels=in_channels,
        features=int(model_config.get("features", 64)),
        num_layers=int(model_config.get("num_layers", 5)),
        official=bool(model_config.get("official", False)),
        nc=model_config.get("nc"),
        nb=int(model_config.get("nb", 4)),
        act_mode=str(model_config.get("act_mode", "R")),
        downsample_mode=str(model_config.get("downsample_mode", "strideconv")),
        upsample_mode=str(model_config.get("upsample_mode", "convtranspose")),
    ).to(device)
    if scheduler is not None:
        raise ValueError(f"Unsupported DRUNet scheduler: {scheduler}. Use None for the sigma25 production protocol.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    lpips_model = create_lpips_model(device)
    history = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": [], "lpips": [], "fid": []}

    print(f"Start training experiment={experiment_name}")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics, sample_noisy, sample_clean, sample_pred = evaluate(model, test_loader, device, criterion, lpips_model)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["psnr"].append(val_metrics["psnr"])
        history["ssim"].append(val_metrics["ssim"])
        history["lpips"].append(val_metrics["lpips"])
        history["fid"].append(val_metrics["fid"])
        print(
            f"[{experiment_name}] epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['val_loss']:.6f} "
            f"psnr={val_metrics['psnr']:.4f} "
            f"ssim={val_metrics['ssim']:.4f} "
            f"lpips={val_metrics['lpips']:.4f} "
            f"fid={val_metrics['fid']:.4f}"
        )

    checkpoint_path = (checkpoint_dir or output_dir) / f"{experiment_name}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[{experiment_name}] saved checkpoint: {checkpoint_path}")
    final_metrics = compute_metrics(sample_pred, sample_clean, lpips_model)
    final_metrics["validation_loss"] = float(history["val_loss"][-1])
    print(f"[{experiment_name}] final sample metrics: {final_metrics}")

    run_config = {
        "experiment_name": experiment_name,
        "mode": mode,
        "filtered_indices_path": str(filtered_indices_path) if filtered_indices_path is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "dataset_name": dataset_name,
        "data_root": str(data_root),
        "download": download,
        "in_channels": in_channels,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer": "Adam",
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "seed": seed,
        "sigma_mode": sigma_mode,
        "fixed_sigma": fixed_sigma,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "num_workers": num_workers,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
        "model_config": model_config,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    save_metrics_table(history, results_dir / "metrics_history.csv")
    save_metrics_table(history, results_dir / "training_history.csv")
    save_loss_curve(history, results_dir / "loss_curve.png", title=f"Loss | {experiment_name}")
    save_visualization(sample_noisy, sample_clean, sample_pred, output_dir / f"{experiment_name}_example.png")
    save_per_image_metrics_and_qualitative(model, test_loader, device, lpips_model, results_dir, experiment_name)

    return {"experiment_name": experiment_name, "output_dir": output_dir, "final_metrics": final_metrics}
