import h5py
import matplotlib.pyplot as plt

# === пути ===
NOISY_PATH = r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\data\dataset_noisy/dataset_noisy.h5"
TOPK_PATH = "filtered_dataset.h5"
QQ_PATH = "filtered_dataset_qq.h5"

N = 10  # сколько картинок показывать

# === загрузка ===
with h5py.File(NOISY_PATH, "r") as f:
    noisy = f["dataset"][:N]

with h5py.File(TOPK_PATH, "r") as f:
    topk = f["dataset"][:N]

with h5py.File(QQ_PATH, "r") as f:
    qq = f["dataset"][:N]

# === визуализация ===
plt.figure(figsize=(12, 4))

for i in range(N):
    # noisy
    plt.subplot(3, N, i + 1)
    plt.imshow(noisy[i], cmap='gray')
    plt.axis('off')

    # top-k
    plt.subplot(3, N, N + i + 1)
    plt.imshow(topk[i], cmap='gray')
    plt.axis('off')

    # QQ
    plt.subplot(3, N, 2*N + i + 1)
    plt.imshow(qq[i], cmap='gray')
    plt.axis('off')

plt.suptitle("Top: Noisy | Middle: Top-k | Bottom: QQ filtering", fontsize=14)
plt.tight_layout()
plt.show()