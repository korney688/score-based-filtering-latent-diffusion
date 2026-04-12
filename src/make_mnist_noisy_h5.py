import numpy as np
import h5py

input_path = r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\data\dataset_clean_mnist\dataset_clean_mnist.h5"
output_path = r"E:\ITMO\НИР Хватов\2 семместр\project\experiments\exp_001\data/dataset_noisy/dataset_noisy.h5"

noise_std = 0.3

with h5py.File(input_path, "r") as f_in:
    x = f_in["dataset"][:].astype(np.float32)   # (N, 28, 28)

noise = np.random.randn(*x.shape).astype(np.float32) * noise_std
x_noisy = x + noise
x_noisy = np.clip(x_noisy, 0.0, 1.0)

signal_power = np.mean(x ** 2, axis=(1, 2)) + 1e-12
noise_power = np.mean(noise ** 2, axis=(1, 2)) + 1e-12
snr = 10 * np.log10(signal_power / noise_power)

with h5py.File(output_path, "w") as f_out:
    f_out.create_dataset("dataset", data=x_noisy.astype(np.float32))
    f_out.create_dataset("snr", data=snr.astype(np.float32))

print("saved:", output_path)
print("shape:", x_noisy.shape, x_noisy.dtype)
print("snr mean:", snr.mean())