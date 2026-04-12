import numpy as np
import h5py
from torchvision import datasets

mnist = datasets.MNIST(root="./data", train=True, download=True)

N = 512
images = []

for i in range(N):
    img = np.array(mnist[i][0]) / 255.0  # [0,1]
    images.append(img.astype(np.float32))

data = np.stack(images, axis=0)  # (N, 28, 28)

with h5py.File("dataset_noisy.h5", "w") as f:
    f.create_dataset("dataset", data=data)

print("MNIST saved:", data.shape)
