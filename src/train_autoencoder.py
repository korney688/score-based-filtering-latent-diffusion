from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from autoencoder import SimpleAE


class CleanMnistDataset(Dataset):
    def __init__(self, h5_path: Path):
        # Загружаем clean MNIST из HDF5 и сразу приводим к float32.
        with h5py.File(h5_path, "r") as h5_file:
            data = h5_file["dataset"][:].astype(np.float32)

        if data.ndim != 3 or data.shape[1:] != (28, 28):
            raise ValueError(f"Expected shape (N, 28, 28), got {data.shape}")

        # Нормализуем значения из [0, 1] в [-1, 1].
        data = data * 2.0 - 1.0

        # Добавляем размерность канала: (N, 1, 28, 28).
        self.data = torch.from_numpy(data[:, None, :, :])

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


def denormalize_to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    # Возвращаем тензор из [-1, 1] обратно в [0, 1] для отображения.
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    example_batch = None
    example_reconstruction = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            # Модель возвращает значения в [0, 1] из-за Sigmoid.
            reconstruction_01 = model(batch)

            # Переводим реконструкцию в [-1, 1], чтобы loss считался
            # в той же шкале, что и нормализованный вход.
            reconstruction = reconstruction_01 * 2.0 - 1.0
            loss = criterion(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

            example_batch = batch.detach().cpu()
            example_reconstruction = reconstruction.detach().cpu()

        mean_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.6f}")

    if example_batch is None or example_reconstruction is None:
        raise RuntimeError("Training did not produce any batches.")

    return model, example_batch, example_reconstruction


def show_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor) -> None:
    # Берем первый объект из последнего батча и переводим его в [0, 1].
    original_img = denormalize_to_unit_interval(original[0, 0]).numpy()
    reconstructed_img = denormalize_to_unit_interval(reconstructed[0, 0]).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(original_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Оригинал")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Реконструкция")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Определяем корень проекта относительно текущего файла.
    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        project_root
        / "experiments"
        / "exp_001"
        / "data"
        / "dataset_clean_mnist"
        / "dataset_clean_mnist.h5"
    )
    model_path = project_root / "models" / "autoencoder.pth"

    # Подготавливаем датасет и загрузчик.
    dataset = CleanMnistDataset(data_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Выбираем устройство: GPU при наличии, иначе CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAE().to(device)

    model, example_batch, example_reconstruction = train(
        model=model,
        loader=loader,
        device=device,
        epochs=30,
        lr=1e-3,
    )

    # Сохраняем веса после обучения.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")

    # Показываем пример оригинала и реконструкции.
    show_reconstruction(example_batch, example_reconstruction)


if __name__ == "__main__":
    main()
