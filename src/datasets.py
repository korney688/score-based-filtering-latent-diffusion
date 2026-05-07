import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class my_dataset(Dataset):
    def __init__(
        self,
        h5_path,
        data_key="dataset",
        snr_key="",
        in_memory=False,
        stats=None,
        apply_log=True,
        apply_norm=True,
        apply_split=True,
        calc_stats_samples=1000,
        data_mode="image",
    ):
        if data_mode != "image":
            raise ValueError(f"Unsupported data_mode for MNIST pipeline: {data_mode}")

        self.h5_path = h5_path
        self.data_key = data_key
        self.snr_key = snr_key
        self.in_memory = in_memory
        self.h5_file = None
        self.dset_data = None
        self.dset_snr = None

        with h5py.File(self.h5_path, "r") as h5_file:
            if self.data_key not in h5_file:
                raise ValueError(f"Key {self.data_key} not found in {self.h5_path}")

            self.shape = h5_file[self.data_key].shape
            self.num_samples = self.shape[0]

            if self.in_memory:
                log.info(f"Loading {self.h5_path} into RAM...")
                self.data = h5_file[self.data_key][:]
                self.snr = h5_file[self.snr_key][:] if self.snr_key in h5_file else None
                log.info("RAM loading complete.")
            else:
                self.data = None
                self.snr = None

    def _ensure_file_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.dset_data = self.h5_file[self.data_key]
            self.dset_snr = self.h5_file[self.snr_key] if self.snr_key in self.h5_file else None

    def __getitem__(self, idx):
        if self.in_memory:
            raw_data = self.data[idx]
            raw_snr = self.snr[idx] if self.snr is not None else 0.0
        else:
            self._ensure_file_open()
            raw_data = self.dset_data[idx]
            raw_snr = self.dset_snr[idx] if self.dset_snr is not None else 0.0

        img = np.asarray(raw_data, dtype=np.float32)
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0)
        data_tensor = torch.from_numpy(img)

        if self.snr_key == "":
            return data_tensor

        snr_tensor = torch.tensor(raw_snr, dtype=torch.float32)
        return data_tensor, snr_tensor

    def __len__(self):
        return self.num_samples

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
