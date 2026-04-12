import torch
import numpy as np
from torch import nn


class DnCNN_3D(nn.Module):
    def __init__(self, channels, num_of_layers=17, num_features=64):
        super(DnCNN_3D, self).__init__()

        kernel_size = 3
        features = num_features
        layers = []

        # Encoder -------------------------------------
        layers.append(
            nn.Conv3d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                padding_mode="circular",
                bias=True,
            )
        )

        layers.append(Cplx_cardioid())

        # Latent space denoising -----------------------
        for i in range(num_of_layers - 2):
            layers.append(
                nn.Conv3d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding="same",
                    # dilation=dill,
                    padding_mode="circular",
                    bias=True,
                )
            )
            layers.append(Cplx_cardioid())

        # Decoder ---------------------------------------
        layers.append(
            nn.Conv3d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode="circular",
                bias=True,
            )
        )

        self.tdncnn = nn.Sequential(*layers)

        self.tdncnn.apply(complex_weight_init)

    def forward(self, x):
        N_batch, N_ue, N_bs, N_f = x.shape
        x = x.view(N_batch, N_ue * 2, 4, 8, N_f)
        out = self.tdncnn(x)
        out = out.view(N_batch, N_ue, N_bs, N_f)
        return out


class Cplx_cardioid(nn.Module):
    def __init__(self):
        super(Cplx_cardioid, self).__init__()

    def forward(self, x):
        phase = torch.angle(x)
        cmod = 0.5 * (1 + torch.cos(phase))
        return cmod * x


def complex_weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        new_weights = torch.empty_like(m.weight, dtype=torch.complex64)
        torch.nn.init.xavier_uniform_(new_weights.real)
        torch.nn.init.xavier_uniform_(new_weights.imag)
        m.weight.data = new_weights
        if m.bias is not None:
            new_bias = torch.zeros_like(m.bias, dtype=torch.complex64)
            m.bias.data = new_bias


def get_device(device=None, quiet=False):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    if not quiet:
        print("Using device: {}".format(device))

    return device

def build_DnCNN_3D_model(device: str, mode='large') -> DnCNN_3D:
    """Инициализация модели"""
    # cfg - на будущее
    if mode=='small':
        model = DnCNN_3D(channels=8,
                     num_of_layers=15,
                     num_features=16)

    elif mode=='large':
        model = DnCNN_3D(channels=8,
                 num_of_layers=15,
                 num_features=64)
    
    return model.to(device)
    
if __name__ == "__main__":
    input_size = (10, 8, 4, 8, 96)
    device = get_device("cpu", False)

    t = torch.rand(input_size, device=device) + 1j * torch.rand(
        input_size, device=device
    )
    model = DnCNN_3D(channels=3, num_of_layers=10, num_features=64).to(device)
    out = model(t).cpu()

    print(out.shape)
    
