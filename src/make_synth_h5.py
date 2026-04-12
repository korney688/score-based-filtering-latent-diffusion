import numpy as np
import h5py

N = 32
D = 4
H = 16
W = 16

data = []

for _ in range(N):
    real = np.random.randn(D, H, W).astype(np.float32)
    imag = np.random.randn(D, H, W).astype(np.float32)
    x = (real + 1j * imag).astype(np.complex64)
    data.append(x)

data = np.stack(data, axis=0)  # (N, D, H, W)

with h5py.File("synthetic.h5", "w") as f:
    f.create_dataset("dataset", data=data)