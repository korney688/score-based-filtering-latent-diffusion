import csv
import json
from pathlib import Path
from urllib.error import URLError

import lpips
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.utils import make_grid, save_image

from src.TDnCNN_image import DnCNN_2D, compute_metrics, evaluate, train_one_epoch
from src.TDnCNN_image import compute_lpips, compute_ssim_item
from src.tdncnn_datasets import create_dataloaders


def create_lpips_model(device: torch.device) -> nn.Module:
    try:
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except (URLError, OSError, RuntimeError) as error:
        print(
            "LPIPS alexnet weights could not be downloaded from torchvision. "
            "Falling back to offline mode with random trunk weights."
        )
        print(f"LPIPS init error: {error}")
        model = lpips.LPIPS(net="alex", pnet_rand=True).to(device)
        model.eval()
        return model


def save_visualization(
    x_noisy: torch.Tensor,
    x_clean: torch.Tensor,
    x_pred: torch.Tensor,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    clean_image = x_clean[0].numpy()
    noisy_image = x_noisy[0].numpy()
    pred_image = x_pred[0].numpy()
    if clean_image.shape[0] == 1:
        clean_image = clean_image[0]
        noisy_image = noisy_image[0]
        pred_image = pred_image[0]
        cmap = "gray"
    else:
        clean_image = clean_image.transpose(1, 2, 0).clip(0.0, 1.0)
        noisy_image = noisy_image.transpose(1, 2, 0).clip(0.0, 1.0)
        pred_image = pred_image.transpose(1, 2, 0).clip(0.0, 1.0)
        cmap = None

    imshow_kwargs = {"cmap": cmap, "vmin": 0.0, "vmax": 1.0} if cmap == "gray" else {}

    axes[0].imshow(clean_image, **imshow_kwargs)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    axes[1].imshow(noisy_image, **imshow_kwargs)
    axes[1].set_title("Noisy")
    axes[1].axis("off")

    axes[2].imshow(pred_image, **imshow_kwargs)
    axes[2].set_title("TDnCNN")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


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


def save_metrics_grid(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    epochs = range(1, len(history["psnr"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["psnr"])
    axes[0, 0].set_title("PSNR")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("psnr")
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["ssim"])
    axes[0, 1].set_title("SSIM")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("ssim")
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, history["lpips"])
    axes[1, 0].set_title("LPIPS")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("lpips")
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, history["fid"])
    axes[1, 1].set_title("FID")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("fid")
    axes[1, 1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


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


def psnr_from_mse(mse_value: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(1.0 / torch.sqrt(mse_value + 1e-8))


def save_error_map_grid(x_pred: torch.Tensor, x_clean: torch.Tensor, output_path: Path, max_items: int = 8) -> None:
    error = (x_pred[:max_items] - x_clean[:max_items]).abs()
    if error.shape[1] == 3:
        error = error.mean(dim=1, keepdim=True)
    max_value = error.flatten(start_dim=1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
    error = error / max_value
    grid = make_grid(error.cpu(), nrow=max(1, min(max_items, error.shape[0])), padding=2)
    save_image(grid, output_path)


@torch.no_grad()
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
    for x_noisy, x_clean in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        x_pred = model(x_noisy).clamp(0.0, 1.0)

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
        save_error_map_grid(qualitative_pred, qualitative_clean, output_dir / "error_maps.png", max_items=max_qualitative_samples)


def run_experiment(
    experiment_name: str,
    mode: str,
    filtered_indices_path: Path | None,
    output_dir: Path,
    data_root: Path,
    checkpoint_dir: Path | None = None,
    dataset_name: str = "mnist",
    download: bool = False,
    in_channels: int = 1,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
    sigma_min: float = 0.1,
    sigma_max: float = 0.8,
    num_workers: int = 0,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = create_dataloaders(
        filtered_indices=filtered_indices_path,
        batch_size=batch_size,
        mode=mode,
        seed=seed,
        data_root=data_root,
        dataset_name=dataset_name,
        download=download,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )

    model = DnCNN_2D(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    lpips_model = create_lpips_model(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "fid": [],
    }

    print(f"Start training experiment={experiment_name}")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics, sample_noisy, sample_clean, sample_pred = evaluate(
            model,
            test_loader,
            device,
            criterion,
            lpips_model,
        )

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
        "seed": seed,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "num_workers": num_workers,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    save_visualization(
        sample_noisy,
        sample_clean,
        sample_pred,
        output_dir / f"{experiment_name}_example.png",
    )
    save_metrics_table(history, results_dir / "metrics_history.csv")
    save_loss_curve(history, results_dir / "loss_curve.png", title=f"Loss | {experiment_name}")
    save_metrics_grid(history, results_dir / "metrics_grid.png", title=f"Metrics | {experiment_name}")
    save_per_image_metrics_and_qualitative(
        model=model,
        loader=test_loader,
        device=device,
        lpips_model=lpips_model,
        output_dir=results_dir,
        run_name=experiment_name,
    )

    return {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "final_metrics": final_metrics,
    }
