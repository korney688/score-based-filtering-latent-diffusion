import csv
from pathlib import Path
from urllib.error import URLError

import lpips
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.TDnCNN_image import DnCNN_2D, compute_metrics, evaluate, train_one_epoch
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

    axes[0].imshow(x_clean[0, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    axes[1].imshow(x_noisy[0, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Noisy")
    axes[1].axis("off")

    axes[2].imshow(x_pred[0, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
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


def run_experiment(
    experiment_name: str,
    mode: str,
    clean_path: Path,
    noisy_path: Path,
    filtered_indices_path: Path | None,
    output_dir: Path,
    batch_size: int = 32,
    split: float = 0.8,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = create_dataloaders(
        clean_path=clean_path,
        noisy_path=noisy_path,
        filtered_indices=filtered_indices_path,
        batch_size=batch_size,
        split=split,
        mode=mode,
        seed=seed,
    )

    model = DnCNN_2D().to(device)
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

    final_metrics = compute_metrics(sample_pred, sample_clean, lpips_model)
    print(f"[{experiment_name}] final sample metrics: {final_metrics}")

    torch.save(model.state_dict(), output_dir / f"{experiment_name}.pth")
    save_visualization(
        sample_noisy,
        sample_clean,
        sample_pred,
        output_dir / f"{experiment_name}_example.png",
    )
    save_metrics_table(history, results_dir / "metrics_history.csv")
    save_loss_curve(history, results_dir / "loss_curve.png", title=f"Loss | {experiment_name}")
    save_metrics_grid(history, results_dir / "metrics_grid.png", title=f"Metrics | {experiment_name}")

    return {
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "final_metrics": final_metrics,
    }
