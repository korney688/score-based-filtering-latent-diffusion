import torch
from typing import Union


def get_noise(H: torch.Tensor, SNR_dB: Union[int, float, torch.Tensor]) -> torch.Tensor:
    """
    Generates AWGN for channel tensor and desired
    signal-to-noise ratio (SNR) level in dB

    Args:
        H : Input channel 4D tensor, with N_batch as 0-th dimension
        SNR_dB : Desired SNR level in decibels.
                 - if scalar/int then same SNR will be used for all samples
                 - if torch.Tensor of len N_batch, then creates noise for each
                   sample correspondingly

    Returns:
        noise: complex-valued (torch.complex64) AWGN tensor of same shape as input 'H'
    """

    if len(H.shape) != 4:
        raise ValueError("H is expected to be a 4D tensor")

    signal_power = H.abs().pow(2).mean((-1, -2, -3))
    SNR_linear = torch.pow(torch.tensor(10), SNR_dB / torch.tensor(10))

    # Noise amplitude
    sigma_noise = (signal_power / SNR_linear) ** 0.5

    # Norm by 1/sqrt(2) is already done by torch
    noise = torch.randn(*H.shape, dtype=torch.complex64)

    return sigma_noise.view(-1, 1, 1, 1) * noise


def get_fixed_power_noise(shape, Pn_sample_dB: Union[int, float, torch.Tensor]) -> torch.Tensor:
    """
    Generates AWGN for channel tensor and desired noise power in dB

    Args:
        H : Input channel 4D tensor, with N_batch as 0-th dimension
        SNR_dB : Desired noise power per sample in decibels.

    Returns:
        noise: complex-valued (torch.complex64) AWGN tensor of same shape as input 'H'
    """
    # Noise amplitude
    sigma_noise = torch.sqrt(torch.pow(torch.tensor(10), Pn_sample_dB/torch.tensor(10)))

    # Norm by 1/sqrt(2) is already done by torch
    noise = torch.randn(shape, dtype=torch.complex64)

    return sigma_noise.view(-1, 1, 1, 1) * noise

def get_precoder(H: torch.Tensor, rank: int, is_subband: bool = True):
    """
    Compute SVD-precoder for specified transmission 'rank'
    The precoder is formed from the dominant right singular vectors of the channel matrices.

    - If is_subband == True (default):
        A single precoder per batch is computed by stacking all
        subcarriers and UE antennas into one tall matrix.
        The resulting W has shape:
            (N_batch, 1, N_bs, rank)

    - If is_subband == False:
        A precoder is computed per subcarrier (and per batch) from
        H[b, f] of shape (N_ue, N_bs).
        The resulting W has shape:
            (N_batch, N_f, N_bs, rank)

    Args:
        H:
            Channel tensor as described above. Preferred shape is (N_batch, N_ue, N_bs, N_f)
        rank:
            Desired precoder rank (number of right singular vectors).
            Must satisfy rank <= N_bs.
        is_subband:
            If True, compute one common precoder per batch by collapsing
            the (N_ue, N_f) dimensions.
            If False, compute a per-subcarrier precoder.
    """

    if len(H.shape) != 4:
        raise ValueError("H is expected to be a 4D tensor")

    H = H.permute(0, 3, 1, 2)

    # To warning
    assert H.shape[2:] == (4, 64), f"Expected H to have last dims with shape of (4, 64). Got {H.shape[2:]}"

    N_bs, N_batch = H.shape[-1], H.shape[0]

    if is_subband:
        # Prepare batch of matrices 1 x N_f*N_ue x N_bs
        H = H.reshape(N_batch, 1, -1, N_bs)

    # Truncated SVD
    _, S, Vh = torch.linalg.svd(H, full_matrices=False)

    # Use conjugated singular vectors as a precoder
    W = Vh[:, :, :rank, :].conj().transpose(-1, -2)
    return W


# Metrics ===================================================================================================
def capacity(H: torch.Tensor, W: torch.Tensor, SNR_dB: float) -> torch.Tensor:
    """
    Calculate MIMO channel SU-rate metric using channel and BS-precoder.

    Args:
        H: Channel tensor to which the precoder should be used.
        W: Precoder tensor.
        SNR_dB: Desired DL signal-to-noise ratio in dB. Should be scalar

    Returns:
        C: (Nb x Nf) Tensor, containing SU-rate per subcarrier and batch.
    """

    if len(H.shape) != 4 and len(W.shape) != 4:
        raise ValueError("H and W is expected to be a 4D tensors")

    # prepare for batched matrix-operations
    H = H.permute(0, 3, 1, 2)  # Nb x N_f x Nue x Nbs

    # Signal power (Nb x 1 x 1 x 1)
    Ps = H.abs().pow(2).mean((-1, -2, -3), keepdim=True)
    SNR = 10 ** (0.1 * SNR_dB)
    # Noise power (Nb x 1 x 1 x 1)
    Pn = Ps / SNR

    # Effective matrix (Nb x Nf x Nue x Rank)
    HW = H @ W

    # Covariance matrix (Nb x Nf x Nue x Nue)
    R = HW @ HW.conj().transpose(-1, -2)

    # Identity (1 x 1 x Nue x Nue)
    I = torch.eye(R.shape[-1], device=R.device)[None, None]

    # Calculate capacity (Nb x Nf)
    C = torch.real(torch.log2(torch.linalg.det(R / Pn + I)))
    return C
