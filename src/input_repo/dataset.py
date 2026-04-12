import numpy as np
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from typing import List, Sequence, Optional, Callable, Tuple, Union
from torch.utils.data import DataLoader
from torch.fft import fft, ifft, fftshift
from torchvision import transforms
import h5py as h5

# Configure h5py for reading complex-valued .mat (v7.3) files
cfg = h5.get_config()
cfg.complex_names = ("real", "imag")


class MultiRoadDataset(Dataset):
    """
    PyTorch Dataset for multiple large .mat (v7.3) files containing channel data.

    Each file is expected to contain a dataset named 'H' with shape:
        (Nue, Nbs, Nf, T) (in fortran order)
    where:
        - T   : number of time samples (ttis)
        - Nf  : number of subcarriers
        - Nbs : number of base station antennas
        - Nue : number of UE antennas

    Important:
        - The files are opened via h5py, and the dataset 'H' is accessed lazily.
          Data is not fully loaded into RAM; only the slice needed for a given
          index is read on demand.
        - __getitem__ returns a tensor of shape (Nue, Nbs, Nf), obtained by
          taking a slice over time (one index along T) and then transposing
          the remaining dimensions.
        - Since all the variables in matlab .mat files are stored in column-wise
          format, and numpy and pytorch expects row-wise format, the size .mat
          file is flipped after reading. Transpose operation is used in __getitem__
          method to transform to expected shape.
    """

    def __init__(self, paths: Sequence[str], transform: Optional[Callable] = None):
        """
        Args:
            paths:
                Sequence of paths to .mat (v7.3) files. Each must contain a
                dataset 'H'.
            transform:
                Optional callable applied to each sample after converting it
                to a torch.Tensor. It should accept and return a tensor.
        """
        super().__init__()

        self.paths = paths
        self.transform = transform

        # Number of time samples T for each file
        self._T_per_file: List[int] = []
        self._shapes = []

        # Load datasets and precomputes per-file lengths along time dim T
        for path in self.paths:
            with h5.File(path, "r") as f:
                arr = f["H"]

                # Expect 4-dim tensor (T, Nf, Nbs, N_ue)
                if arr.ndim != 4:
                    raise ValueError(
                        f"File '{path}' must have 4 dimensions (T, Nf, Nbs, N_ue), got {arr.shape}"
                    )

                T, Nf, Nbs, Nue = arr.shape

                if T <= 0:
                    raise ValueError(
                        f"File '{path}' must have at least one time sample"
                    )

                self._T_per_file.append(T)
                self._shapes.append((T, Nf, Nbs, Nue))

        # Cumulative time lengths, used to map a global index to (file_idx, local_idx).
        # Example:
        #   _T_per_file      = [100, 200, 50]
        #   _cum_T_per_file  = [0, 100, 300, 350]
        self._cum_T_per_file = np.cumsum([0] + self._T_per_file)

        # Total number of time samples across all files.
        self._total_samples = int(self._cum_T_per_file[-1])

        self._files = [None] * len(self.paths)
        self._datasets = [None] * len(self.paths)

    def __len__(self) -> int:
        """Return the total number of time samples across all files."""
        return self._total_samples

    def _locate_index(self, idx: int) -> Tuple[int, int]:
        """
        Map a global index to (file_idx, local_idx).

        Args:
            idx: Global index in range [0, _total_samples).

        Returns:
            file_idx: Index of the file in self._arrays.
            local_idx: Index inside that file along the time dimension T.
        """
        file_idx = bisect_right(self._cum_T_per_file, idx) - 1
        local_idx = idx - self._cum_T_per_file[file_idx]
        return file_idx, local_idx

    def _get_dataset(self, file_idx: int) -> h5.Dataset:
        dset = self._datasets[file_idx]
        if dset is None:
            f = h5.File(self.paths[file_idx], "r")
            self._files[file_idx] = f
            dset = f["H"]
            self._datasets[file_idx] = dset
        return dset

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample corresponding to a time index across all files.

        Args:
            idx: Global time index. Negative indices are supported
                 (Python-style indexing).

        Returns:
            A torch.Tensor of shape (Nue, Nbs, Nf), obtained from:
                sel_data = H[local_idx].transpose()
        """
        # Support negative indexing
        if idx < 0:
            idx = self._total_samples + idx

        if idx >= self._total_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {self._total_samples}"
            )

        # Find which file this index belongs to, and the local index within that file
        file_idx, local_idx = self._locate_index(idx)

        # Select the corresponding h5py dataset.
        sel_array = self._get_dataset(file_idx)

        # Read channel sample for one TTi. Transpose to move from col-wise into row-wise format
        sel_data = sel_array[local_idx].transpose()

        # TODO: Fix order in matlab
        N_ue, Nbs, N_f = sel_data.shape
        sel_data = np.reshape(sel_data, (N_ue, 2, 4, 8, N_f), order="F").reshape(
            N_ue, Nbs, N_f
        )
        sel_data = torch.from_numpy(sel_data)
        # sel_data = sel_data.type(torch.complex64)

        if self.transform is not None:
            sel_data = self.transform(sel_data)

        # sample normalisation
        # don't use it for real-world simulations
        #sel_data = sel_data / torch.linalg.norm(sel_data) # tranfer to gen_noisy_dataset

        return sel_data

    def __del__(self):
        """
        Close all h5 files if Dataset has been destroyed
        """
        if hasattr(self, "_files"):
            for f in self._files:
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass


# Transforms =========================================================
class Antenna2Beam(object):
    """
    Transform signal from the BS antenna domain into the beam domain.

    This uses Kronecker products of DFT matrices (horizontal and vertical)
    and polarization identity to build a unitary transform matrix B:
        B = kron(kron(I_pol, F_v), F_h)

    The beam domain is typically much sparser than the antenna domain.

    Expected input sample shape:
        (Nue, Nbs, Nf)
    where:
        - Nue : number of UE antennas
        - Nbs : number of BS antennas (must equal N_pol * N_hor * N_ver)
        - Nf  : number of subcarriers / frequency bins
    """

    def __init__(
        self, N_pol: int, N_hor: int, N_ver: int, inverse: bool = False
    ) -> None:
        """
        Args:
            N_pol: Number of polarization components (2).
            N_hor: Number of horizontal elements in the BS array (8).
            N_ver: Number of vertical elements in the BS array (4).
            inverse: If True, apply the inverse (conjugate) DFT transform.
        """
        self.N_pol = N_pol
        self.N_hor = N_hor
        self.N_ver = N_ver
        self.inverse = inverse

        # Horizontal and vertical DFT matrices
        Fh = self._dft_mtx(self.N_hor)  # N_hor x N_hor
        Fv = self._dft_mtx(self.N_ver)  # N_ver x N_ver

        # Polarization identity
        Ip = torch.eye(self.N_pol)  # N_pol x N_pol

        # Inverse transform from beams into BS antenna
        if inverse:
            Fh, Fv = Fh.conj(), Fv.conj()

        # Full beamforming matrix (N_bs x N_bs)
        self.B = torch.kron(torch.kron(Ip, Fv), Fh)

    def __call__(self, sample):
        Nue, Nbs, Nf = sample.shape
        assert Nbs == self.N_pol * self.N_hor * self.N_ver

        # reorder for batch multiplication, multiplication, inverse reordering
        return torch.permute(sample.permute(0, 2, 1) @ self.B, (0, 2, 1))

    def _dft_mtx(self, X):
        """
        Args:
            X: Matrix size
        Returns:
            DFT matrix of shape (X, X) with unitary scaling
        """
        return torch.fft.fft(torch.eye(X), norm="ortho")


class Freq2Delay(object):
    """
    Transform along the last dimension between frequency and delay domains
    using orthonormal FFT/IFFT.

    - If inverse=False  : frequency -> delay via IFFT
    - If inverse=True   : delay -> frequency via FFT
    """

    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, sample):
        if self.inverse:
            # Delay -> frequency domain (FFT)
            return torch.fft.fft(sample, dim=-1, norm="ortho")
        else:
            # Frequency -> delay domain (IFFT)
            return torch.fft.ifft(sample, dim=-1, norm="ortho")


class HardWindow(object):
    """
    Simple "hard window" operation that selects a fixed number of delays from
    the left and right ends of the last dimension and concatenates them.
    Can be used only after Freq2Delay transform!

    Given last-dim length N delay domain signal, it takes:
        - 'right' delay bins from the end       -> sample[..., -right:]
        - 'left'  delay bins from the beginning -> sample[..., :left]
    and concatenates [right, left] along the last dimension.
    """

    def __init__(self, left: int = 13, right: int = 3):
        self.left = left
        self.right = right

    def __call__(self, sample):
        right = sample[..., -self.right :]
        left = sample[..., : self.left]
        sample = torch.concatenate(tensors=(right, left), dim=-1)
        return sample


class SubbandSelect(object):
    """
    Select a single subband from the subcarrier dimension.

    The input is assumed to have shape (Nue, Nbs, Nf). The last dimension Nf
    is split into N_subbands equal subbands, and one of them is selected.

    Selection is done either:
        - by a fixed index (method is int), or
        - randomly at each call (method == 'random').
    """

    def __init__(self, N_subbands: int, method: Union[int, str]):
        self.N_subbands = N_subbands
        self.method = method

    def __call__(self, sample):
        Nue, Nbs, Nf = sample.shape
        assert (
            Nf % self.N_subbands == 0
        ), "Cannot reshape: Nf must be divisible by N_subbands"

        # Reshape to (Nue, Nbs, N_subbands, Nf_per_subband)
        sample = sample.reshape(Nue, Nbs, self.N_subbands, -1)

        # Fixed subband index
        if isinstance(self.method, int):
            return sample[:, :, self.method, :]

        # Random subband index per call
        elif self.method == "random":
            idx = np.random.randint(low=0, high=self.N_subbands)
            return sample[:, :, idx, :]

        else:
            raise NotImplementedError(
                f"method can be int or 'random', got {self.method}"
            )


if __name__ == "__main__":
    pathlist = ["/Users/albly/GitHub/MIMO_road_generator/data/channel_seed_106.mat"]

    transforms = transforms.Compose(
        [
            SubbandSelect(N_subbands=17, method="random"),
            Freq2Delay(inverse=False),
            HardWindow(),
            Antenna2Beam(N_pol=2, N_hor=8, N_ver=4, inverse=False),
        ]
    )

    dataset = MultiRoadDataset(paths=pathlist, transform=transforms)

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        pin_memory=True,
    )

    for batch in loader:
        print(batch.shape)

        pass
