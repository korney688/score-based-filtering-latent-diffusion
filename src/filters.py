import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

# Создаем логгер для этого файла
log = logging.getLogger(__name__)

def top_k_filter(scores, pcnt):
    log.info(f"Запуск фильтрации Top-K: оставляем {pcnt}% с наименьшим score.")
    
    if len(scores) == 0:
        log.error("В top_k_filter передан пустой список scores!")
        return torch.tensor([])

    N_samples = int(pcnt * scores.shape[0] / 100)
    indices = torch.topk(scores, N_samples, largest=False).indices 
    
    log.info(f"Top-K фильтрация завершена. Отобрано {len(indices)} индексов.")
    return indices


def QQ_spread_filter(scores, pcnt):
    log.info(f"Запуск фильтрации QQ_spread: оставляем {pcnt}% (стратифицированно).")

    scores_np = scores.cpu().numpy()
    
    if len(scores_np) == 0:
        log.error("В QQ_spread_filter передан пустой список scores!")
        return torch.tensor([])

    min_points_per_bin = 30
    
    # Вычисляем количество bins
    num_bins = max(1, len(scores_np) // min_points_per_bin)
    log.debug(f"QQ_spread: разбиение на {num_bins} бинов.")
    
    # Квантили для границ bins
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(scores_np, quantiles)
    
    # Присваиваем каждой точке номер бина
    bin_assignments = np.digitize(scores_np, bin_edges[:-1], right=False) - 1
    bin_assignments = np.clip(bin_assignments, 0, num_bins - 1)
    
    random_indices = []
    
    for i in range(num_bins):
        bin_mask = (bin_assignments == i)
        indices_in_bin = np.where(bin_mask)[0]
        
        if len(indices_in_bin) > 0:
            N = max(1, int(pcnt * len(indices_in_bin) / 100))
            selected = np.random.choice(indices_in_bin, size=min(N, len(indices_in_bin)), replace=False)
            random_indices.extend(selected)
    
    final_indices = torch.from_numpy(np.array(random_indices)).long()
    
    log.info(f"QQ_spread фильтрация завершена. Отобрано {len(final_indices)} индексов.")
    return final_indices


def run_DDPM_filter(dataset, DDPM_model, device, mode='top_k', percent=50):

    log.info(f"Начало отбора данных. Метод: {mode}, Keep: {percent}%")
    
    if len(dataset) == 0:
        log.error("Передан пустой датасет! Фильтрация невозможна.")
        return None

    loader = DataLoader(dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True if device == 'cuda' else False
        )
    DDPM_model.eval()
    
    score_list = []
    
    try:
        # Параметры логирования
        total_batches = len(loader)
        log_interval = max(1, int(total_batches * 0.2)) # каждые 20%
        
        with torch.no_grad():
            for i, x_batch in enumerate(loader):
                x_batch = x_batch.to(device)
            
                t = torch.zeros(len(x_batch), device=device).long() 
                
                current_score = DDPM_model.get_score(x_batch, t)
                
                if torch.isnan(current_score).any():
                    log.error(f"NaN значения обнаружены в score на батче {i}!")
                
                current_score = current_score.reshape(current_score.shape[0], -1)
                score_norm = current_score.norm(dim=1)
                
                score_list.append(score_norm.cpu())

                # Логируем процесс
                if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
                    percent_done = int((i + 1) / total_batches * 100)
                    log.info(f"Обработано: {i + 1}/{total_batches} ({percent_done}%)")


    except Exception as e:
        log.error(f"Ошибка во время цикла расчета score: {e}")
        raise e

    score_norm = torch.cat(score_list, dim=0).detach()
    log.info(f"Расчет score завершен. Всего элементов: {score_norm.shape[0]}")

    if mode == 'top_k':
        return top_k_filter(score_norm, percent)

    elif mode == 'QQ_spread':
        return QQ_spread_filter(score_norm, percent)

    else:
        log.error(f"Передан неизвестный режим фильтрации: {mode}")
        raise ValueError(f"Unknown filtering mode: {mode}")


def run_DDPM_filter_v2(dataset, DDPM_model, device, mode='top_k', percent=50, params=[0,0,1]):

    log.info(f"Начало отбора данных. Метод: {mode}, Keep: {percent}%")
    
    if len(dataset) == 0:
        log.error("Передан пустой датасет! Фильтрация невозможна.")
        return None

    loader = DataLoader(dataset, 
            batch_size=32, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True if device == 'cuda' else False
        )
    DDPM_model.eval()
    
    score_list = []
    
    try:

        log.info(f"Начало расчета score для t с {params[0]} до {params[1]}, количество {params[2]}")
        
        score_list_comm = []
        with torch.no_grad():
            for i in set(np.linspace(params[0], params[1], params[2], dtype='int8')):
                score_list = []
                for x_batch in loader:
                    x_batch = x_batch.to(device)
                    t = torch.zeros(len(x_batch), device=device).int() + i
                    current_score = DDPM_model.get_score(x_batch, t)
                    
                    # if torch.isnan(current_score).any():
                    #     log.error(f"NaN значения обнаружены в score на батче {i}!")
                        
                    current_score = current_score.reshape(current_score.shape[0], -1)
                    score_norm = current_score.norm(dim=1)
                    
                    score_list.append(score_norm)
            
                score_norm = torch.cat(score_list, dim=0)
                score_list_comm.append(score_norm)
                log.info(f"Обработано t = {i}")
        
        score_norm = torch.stack(score_list_comm).norm(dim=0)

    except Exception as e:
        log.error(f"Ошибка во время цикла расчета score: {e}")
        raise e
        
    log.info(f"Расчет score завершен. Всего элементов: {score_norm.shape[0]}")

    if mode == 'top_k':
        return top_k_filter(score_norm, percent)

    elif mode == 'QQ_spread':
        return QQ_spread_filter(score_norm, percent)

    else:
        log.error(f"Передан неизвестный режим фильтрации: {mode}")
        raise ValueError(f"Unknown filtering mode: {mode}")