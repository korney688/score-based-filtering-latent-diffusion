import lpips
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance


class DnCNN_2D(nn.Module):
    def __init__(self, in_channels: int = 1, num_layers: int = 8, features: int = 64):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.append(nn.Conv2d(features, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_pred = self.net(x)
        return x - noise_pred


def repeat_to_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    return x.repeat(1, 3, 1, 1)


def prepare_for_lpips(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    x = x * 2.0 - 1.0
    x = repeat_to_three_channels(x)
    x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    return x


def prepare_for_fid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    x = repeat_to_three_channels(x)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x


def compute_lpips(x_pred: torch.Tensor, x_true: torch.Tensor, lpips_model: nn.Module) -> float:
    pred_lpips = prepare_for_lpips(x_pred)
    true_lpips = prepare_for_lpips(x_true)
    value = lpips_model(pred_lpips, true_lpips)
    return float(value.mean().item())


def create_fid_metric(device: torch.device):
    try:
        fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        return fid
    except ModuleNotFoundError as error:
        print(
            "FID is disabled because torch-fidelity is not installed. "
            "Install `torch-fidelity` or `torchmetrics[image]` to enable it."
        )
        print(f"FID init error: {error}")
        return None


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for x_noisy, x_clean in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)

        x_pred = model(x_noisy)
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

    for x_noisy, x_clean in loader:
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)

        x_pred = model(x_noisy)
        loss = criterion(x_pred, x_clean)

        batch_size = x_noisy.shape[0]
        total_loss += loss.item() * batch_size

        mse_per_item = ((x_pred - x_clean) ** 2).flatten(1).mean(dim=1)
        psnr_per_item = 20 * torch.log10(1.0 / torch.sqrt(mse_per_item + 1e-8))
        total_psnr += float(psnr_per_item.sum().item())

        total_lpips += compute_lpips(x_pred, x_clean, lpips_model) * batch_size

        if fid is not None:
            x_pred_fid = prepare_for_fid(x_pred)
            x_clean_fid = prepare_for_fid(x_clean)
            fid.update(x_clean_fid, real=True)
            fid.update(x_pred_fid, real=False)

        x_pred_np = x_pred.detach().cpu().numpy()
        x_clean_np = x_clean.detach().cpu().numpy()
        for idx in range(batch_size):
            total_ssim += float(
                ssim(
                    x_clean_np[idx, 0],
                    x_pred_np[idx, 0],
                    data_range=1.0,
                )
            )

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


def compute_metrics(
    x_pred: torch.Tensor,
    x_clean: torch.Tensor,
    lpips_model: nn.Module,
) -> dict[str, float]:
    mse = ((x_pred - x_clean) ** 2).mean().item()
    psnr = 20 * torch.log10(1.0 / torch.sqrt(((x_pred - x_clean) ** 2).mean() + 1e-8)).item()

    clean_np = x_clean[0, 0].cpu().numpy()
    pred_np = x_pred[0, 0].cpu().numpy()
    ssim_val = ssim(clean_np, pred_np, data_range=1.0)
    lpips_val = compute_lpips(x_pred, x_clean, lpips_model)

    fid = create_fid_metric(torch.device("cpu"))
    if fid is not None:
        fid.update(prepare_for_fid(x_clean), real=True)
        fid.update(prepare_for_fid(x_pred), real=False)
        fid_val = float(fid.compute().item())
        fid.reset()
    else:
        fid_val = float("nan")

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": float(ssim_val),
        "lpips": lpips_val,
        "fid": fid_val,
    }
