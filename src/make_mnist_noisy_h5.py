from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


INPUT_KEY = "dataset"
NOISE_STD_MIN = 0.1
NOISE_STD_MAX = 0.8


def load_clean_dataset(input_path: Path) -> np.ndarray:
    with h5py.File(input_path, "r") as h5_file:
        clean = h5_file[INPUT_KEY][:].astype(np.float32)

    if clean.ndim != 3 or clean.shape[1:] != (28, 28):
        raise ValueError(
            f"Expected dataset shape (N, 28, 28), got {clean.shape}"
        )

    return clean


def make_noisy_dataset(
    clean: np.ndarray,
    noise_std_min: float = NOISE_STD_MIN,
    noise_std_max: float = NOISE_STD_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng()

    noise_std = rng.uniform(
        noise_std_min,
        noise_std_max,
        size=clean.shape[0],
    ).astype(np.float32)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std[:, None, None],
        size=clean.shape,
    ).astype(np.float32)

    noisy = np.clip(clean + noise, 0.0, 1.0).astype(np.float32)

    signal_power = np.mean(clean**2, axis=(1, 2), dtype=np.float32) + 1e-12
    noise_power = np.mean(noise**2, axis=(1, 2), dtype=np.float32) + 1e-12
    snr = (10.0 * np.log10(signal_power / noise_power)).astype(np.float32)

    return noisy, noise_std, snr


def save_dataset(
    output_path: Path,
    noisy: np.ndarray,
    noise_std: np.ndarray,
    snr: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("dataset", data=noisy, dtype=np.float32)
        h5_file.create_dataset("noise_std", data=noise_std, dtype=np.float32)
        h5_file.create_dataset("snr", data=snr, dtype=np.float32)


def print_diagnostics(noisy: np.ndarray, noise_std: np.ndarray, snr: np.ndarray) -> None:
    print("shape:", noisy.shape)
    print("dtype:", noisy.dtype)
    print("min/max:", float(noisy.min()), float(noisy.max()))
    print("mean noise_std:", float(noise_std.mean()))
    print("first noise_std:", noise_std[:10])
    print("mean snr:", float(snr.mean()))


def show_examples(clean: np.ndarray, noisy: np.ndarray, noise_std: np.ndarray) -> None:
    low_noise_idx = int(np.argmin(noise_std))
    high_noise_idx = int(np.argmax(noise_std))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(clean[low_noise_idx], cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    axes[1].imshow(noisy[low_noise_idx], cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Noisy low\nstd={noise_std[low_noise_idx]:.3f}")
    axes[1].axis("off")

    axes[2].imshow(noisy[high_noise_idx], cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Noisy high\nstd={noise_std[high_noise_idx]:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = (
        project_root
        / "experiments"
        / "exp_001"
        / "data"
        / "dataset_clean_mnist"
        / "dataset_clean_mnist.h5"
    )
    output_path = (
        project_root
        / "experiments"
        / "exp_001"
        / "data"
        / "dataset_noisy_var"
        / "dataset_noisy_var.h5"
    )

    clean = load_clean_dataset(input_path)
    noisy, noise_std, snr = make_noisy_dataset(clean)

    save_dataset(output_path, noisy, noise_std, snr)
    print_diagnostics(noisy, noise_std, snr)
    print("saved:", output_path)

    show_examples(clean, noisy, noise_std)


if __name__ == "__main__":
    main()
