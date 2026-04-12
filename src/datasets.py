import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)

class my_dataset(Dataset):
    def __init__(self, 
                 h5_path, 
                 data_key="dataset", 
                 snr_key='',
                 in_memory=False,
                 stats=None,
                 apply_log=True, 
                 apply_norm=True, 
                 apply_split=True,
                 calc_stats_samples=1000,
                 data_mode="signal"):
        """
        Args:
            h5_path (str): Путь к .h5 файлу.
            data_key (str): Ключ датасета с данными.
            snr_key (str): Ключ датасета с метками SNR (если есть).
            in_memory (bool): Если True, загружает всё в RAM.
            stats (dict): Словарь со статистиками (mean/std). Если None, вычисляются автоматически.
            apply_log (bool): Применять логарифмирование амплитуды.
            apply_norm (bool): Применять нормализацию (StandardScaler).
            apply_split (bool): Разделять на Амплитуду/Фазу (Complex -> 2 Channels).
            calc_stats_samples (int): Количество сэмплов для подсчета статистики.
        """
        
        super().__init__()
        self.h5_path = h5_path
        self.data_key = data_key
        self.snr_key = snr_key
        
        self.in_memory = in_memory
        self.apply_log = apply_log
        self.apply_norm = apply_norm
        self.apply_split = apply_split

        self.h5_file = None
        self.dset_data = None
        self.dset_snr = None

        self.data_mode = data_mode # адаптационный режим  для изображения

        # Считывание параметров
        with h5py.File(self.h5_path, 'r') as f:

            # Проверка наличия ключей
            if self.data_key not in f:
                raise ValueError(f"Ключ {self.data_key} не найден в {self.h5_path}")

            # Основные размерности
            self.shape = f[self.data_key].shape
            self.num_samples = self.shape[0]

            # Если загружаем весь датасет в RAM
            if self.in_memory:
                log.info(f"Загрузка {self.h5_path} в RAM...")
                self.data = f[self.data_key][:]
                if self.snr_key in f:
                    self.snr = f[self.snr_key][:]
                else:
                    self.snr = None
                log.info("Загрузка в RAM завершена.")
            else:
                self.data = None
                self.snr = None

        # Вычисление статистик, если нужна нормализация
        if self.data_mode == "signal" and self.apply_norm and self.apply_split:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = self._compute_global_stats(calc_stats_samples)
        else:
            self.stats = None

        if self.data_mode == "image":
            self.stats = None

    def _ensure_file_open(self):
        """
        Открывает файл, если он еще не открыт (для num_workers > 0)
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.dset_data = self.h5_file[self.data_key]
            if self.snr_key in self.h5_file:
                self.dset_snr = self.h5_file[self.snr_key]
            else:
                self.dset_snr = None

    def _compute_global_stats(self, n_samples):
        """
        Расчет cтатистик для нормализации
        """
        
        with h5py.File(self.h5_path, 'r') as f:
            dset = f[self.data_key]
            total_len = dset.shape[0]
            
            n_samples = min(n_samples, total_len)
            log.info(f"Вычисляем глобальные статистики по {n_samples} сэмплам из HDF5...")
            
            # Выбираем случайные индексы и сортируем
            indices = np.sort(np.random.choice(total_len, n_samples, replace=False))
            
            subset = dset[indices]
            subset = subset.view(np.complex64)

            # Аспоитуда и фаза
            amplitude = np.abs(subset)
            phase = np.angle(subset)

            # Лог шкала по амплитуде
            if self.apply_log:
                log_amplitude = np.log(amplitude + 1e-12)
            else:
                log_amplitude = amplitude

            # Статистики
            stats = {
                'amp_mean': float(log_amplitude.mean()),
                'amp_std':  float(log_amplitude.std()),
                'phase_mean': float(phase.mean()),
                'phase_std':  float(phase.std())
            }
            
            log.info(f"Статистики подсчитаны: {stats}")
            return stats

    def __getitem__(self, idx):
        
        # Получаем данные
        if self.in_memory:
            raw_data = self.data[idx]
            raw_snr = self.snr[idx] if self.snr is not None else 0.0
        else:
            self._ensure_file_open()
            raw_data = self.dset_data[idx]
            raw_snr = self.dset_snr[idx] if self.dset_snr is not None else 0.0

        # Проверки на косплексность и массив
        if not isinstance(raw_data, np.ndarray):
            raw_data = np.array(raw_data)
        if self.data_mode == "signal":   # Адаптация под режим изображения
            if not np.iscomplexobj(raw_data):
                raw_data = raw_data.view(np.complex64)

        # Режим для изображений
        if self.data_mode == "image":
            if not isinstance(raw_data, np.ndarray):
                raw_data = np.array(raw_data)

            img = raw_data.astype(np.float32)

            # нормализация [0,1] → [-1,1]
            img = (img - 0.5) / 0.5

            # [H, W] → [1, H, W]
            img = np.expand_dims(img, axis=0)

            return torch.from_numpy(img)

        # Препроцессинг
        if self.apply_split:

            # Выделение каналов
            amplitude = np.abs(raw_data)
            phase = np.angle(raw_data)

            # Логарифмирование
            if self.apply_log:
                log_amplitude = np.log(amplitude + 1e-12)
            else:
                log_amplitude = amplitude
                
            # Нормализация
            if self.apply_norm and self.stats is not None:
                log_amplitude = (log_amplitude - self.stats['amp_mean']) / (self.stats['amp_std'] + 1e-8)
                phase = (phase - self.stats['phase_mean']) / (self.stats['phase_std'] + 1e-8)
    
            # Сборка тензора (добавляем каналы асплитуды и фазы)
            data_np = np.stack([log_amplitude, phase], axis=-1).astype(np.float32)
            
            # numpy -> tensor
            data_tensor = torch.from_numpy(data_np)

            # Permute к нужной струткуре
            if len(data_tensor.shape) == 4:
                data_tensor = data_tensor.permute(3, 0, 1, 2)
            elif len(data_tensor.shape) == 5:
                data_tensor = data_tensor.permute(0, 4, 1, 2, 3)

        else:
            # numpy -> tensor
            data_tensor = torch.tensor(raw_data)

        # Обработака SNR
        snr_tensor = torch.tensor(raw_snr, dtype=torch.float32)

        # SNR выводим только если требуется
        if self.snr_key == '':
            return data_tensor
        else:
            return data_tensor, snr_tensor
            

    def __len__(self):
        return self.num_samples
    
    def close(self):
        """Явное закрытие"""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None