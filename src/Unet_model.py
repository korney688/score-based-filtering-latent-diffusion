import torch
import torch.nn as nn
import torch.nn.functional as F
import math


aniso_kernel = (1, 3, 3) 
aniso_stride = (1, 2, 2)
    
kernels = [aniso_kernel, aniso_kernel, aniso_kernel]
strides = [aniso_stride, aniso_stride, aniso_stride]

# --- Инициализация весов  ---
def zero_module(module):

    for p in module.parameters():
        p.detach().zero_()
    return module



# --- time_emb ---
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    device = timesteps.device
    
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



# --- ResBlock ---
class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, kernel_size, residual=False, groups=32):
        super().__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2
        else:
            padding = tuple((k - 1) // 2 for k in kernel_size)

        gn_groups = min(groups, in_channels // 4) if in_channels > 4 else 1 
        
        self.norm1 = nn.GroupNorm(gn_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        gn_groups_out = min(groups, out_channels // 4) if out_channels > 4 else 1
        self.norm2 = nn.GroupNorm(gn_groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        self.conv2 = zero_module(self.conv2)

        if self.in_channels != self.out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, time_emb):
        h = x
        
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        t_vec = self.time_emb_proj(time_emb)
        t_vec = t_vec[:, :, None, None]
        h = h + t_vec 

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return self.skip_proj(x) + h


# # --- SelfAttention ---
# class _SelfAttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
        
#         self.attn_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
#         self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        
#     def forward(self, x):
#         b, c, h, w, z = x.shape
#         inp_attn = x.reshape(b, c, h*w*z)
#         inp_attn = self.attn_norm(inp_attn)
#         inp_attn = inp_attn.transpose(1, 2)
#         out_attn, _ = self.mha(inp_attn, inp_attn, inp_attn)
#         out_attn = out_attn.transpose(1, 2).reshape(b, c, h, w, z)
#         return x + out_attn



# --- EncoderBlock ---
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, kernel_size, stride, residual=False):
        super().__init__()
        self.block = _ResBlock(in_channels, out_channels, time_dim, kernel_size, residual=residual)
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x, time_emb):
        x = self.block(x, time_emb)
        y = self.max_pool(x)
        return y, x




# --- DecoderBlock ---
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, kernel_size, stride, residual=False):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.block = _ResBlock(in_channels, out_channels, time_dim, kernel_size, residual=residual)

    def forward(self, x, y, time_emb):
        x = self.transpose(x)

        diffH = y.size(2) - x.size(2)
        diffW = y.size(3) - x.size(3)
        x = F.pad(
            x,
            [diffW // 2, diffW - diffW // 2,
             diffH // 2, diffH - diffH // 2])

        u = torch.cat([x, y], dim=1)
        u = self.block(u, time_emb)
        return u



# --- Bottleneck ---
class _Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, kernel_size, residual=False):
        super().__init__()
        self.block = _ResBlock(in_channels, out_channels, time_dim, kernel_size, residual=residual)
        # self.att = _SelfAttentionBlock(out_channels) 

    def forward(self, x, time_emb):
        x = self.block(x, time_emb)
        #x = self.att(x)
        return x


# --- Unet ---
class UNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 base_dim=64, 
                 time_dim=256, 
                 residual=False,
                 kernel_sizes=[(1,3,3), (1,3,3)], 
                 strides=[(1,2,2), (1,2,2)]):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Глубина сети определяется длиной списка strides
        deep_lvl = len(strides)
        self.features = [base_dim * (2**i) for i in range(deep_lvl)] 
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # --- Encoder ---
        current_in = in_channels
        for idx, (feature, k_size, stride) in enumerate(zip(self.features, kernel_sizes, strides)):
            is_residual = residual if idx > 0 else False
            
            self.encoders.append(
                _EncoderBlock(
                    in_channels=current_in, 
                    out_channels=feature, 
                    time_dim=time_dim, 
                    kernel_size=k_size, 
                    stride=stride, 
                    residual=is_residual
                )
            )
            current_in = feature
            
        # --- Bottleneck ---
        last_kernel = kernel_sizes[-1]
        self.bottleneck = _Bottleneck(
            in_channels=self.features[-1], 
            out_channels=self.features[-1] * 2, 
            time_dim=time_dim,
            kernel_size=last_kernel,
            residual=residual
        )

        # --- Decoder ---
        reversed_features = list(reversed(self.features))
        reversed_kernels = list(reversed(kernel_sizes))
        reversed_strides = list(reversed(strides))

        for idx, feature in enumerate(reversed_features):
            k_size = reversed_kernels[idx]
            stride = reversed_strides[idx]
            is_residual = residual if idx < (deep_lvl - 1) else False

            self.decoders.append(
                _DecoderBlock(
                    in_channels=feature * 2, 
                    out_channels=feature, 
                    time_dim=time_dim, 
                    kernel_size=k_size, 
                    stride=stride, 
                    residual=is_residual
                )
            )

        # --- Out ---
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x, timesteps):
        skip_connections = []
        time_emb = timestep_embedding(timesteps, dim=self.time_dim)

        # --- Encoder ---
        for encoder in self.encoders:
            x, skip = encoder(x, time_emb)
            skip_connections.append(skip)

        # --- Bottleneck ---
        x = self.bottleneck(x, time_emb)

        # --- Decoder ---
        skip_connections = skip_connections[::-1]
        for idx, decoder in enumerate(self.decoders):
            skip = skip_connections[idx]
            x = decoder(x, skip, time_emb) 
            
        return self.final_conv(x)
