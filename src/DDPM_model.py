import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Unet_model import UNet


class DDPM(nn.Module):

    def __init__(self, NN_model, n_steps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__()
        
        self.n_steps = n_steps
        self.device = device
        self.model = NN_model.to(device)
        
        # Линейное расписание для шума
        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(self, x_0, t, noise=None):
        """
        Прямой процесс диффузии: добавляем шум к данным.
        
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t
    
    def get_score(self, x, t):
        """
        Вычисляет score function: ∇log p(x_t).
        
        """

        # Случайцный шум
        noise = torch.randn_like(x)
        
        # Зашумляем данные
        x_t = self.q_sample(x, t, noise)
        
        # Предсказываем шум с помощью UNet
        noise_pred = self.model(x_t, t)
        
        # Извлекаем alpha_bar для соответствующих временных шагов
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        # Score function:
        score = -noise_pred / torch.sqrt(1 - alpha_bar_t)
        
        return score
    
    def train_step(self, x_0):
        """
        Один шаг обучения.
        
        """
        batch_size = x_0.shape[0]
        x_0 = x_0.to(self.device)
        
        # Случайные временные шаги для каждого примера в батче
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        # Генерируем шум
        noise = torch.randn_like(x_0)
        
        # Получаем зашумленные данные
        x_t = self.q_sample(x_0, t, noise)
        
        # Предсказываем шум
        noise_pred = self.model(x_t, t)
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss



def build_DDPM_model(base_dim: int=16, deep: int=3, device: str='cpu') -> DDPM:
    """Инициализация UNet и DDPM"""
    
    # Параметры модели
    aniso_kernel = (1, 3, 3) 
    aniso_stride = (1, 2, 2)
        
    kernels = []
    strides = []

    for _ in range(deep):
        kernels.append(aniso_kernel)
        strides.append(aniso_stride)

    # Инициализация Unet
    NN_model = UNet(in_channels=1,
                   out_channels=1,
                   base_dim=base_dim,
                   time_dim=128,
                   residual=True,
                   kernel_sizes=kernels,
                   strides=strides
                  )
    
    # Инициализация DDPM
    # DDPM_model = DDPM(NN_model,
    #             n_steps=10000,
    #             beta_start=1e-4,
    #             beta_end=0.015,
    #             device=device
    #            )

    DDPM_model = DDPM(NN_model,
            n_steps=10000,
            beta_start=1e-4,
            beta_end=0.02,
            device=device
           )
    
    return DDPM_model.to(device)

