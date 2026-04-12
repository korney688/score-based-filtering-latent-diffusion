## Струткура
```
├── configs/                        # Конфигурационные файлы (Hydra)
│   ├── epochs_evaluation/
│   │   └── default.yaml            # Конфиг eval pipe
│   ├── filter_dataset/
│   │   └── default.yaml            # Конфиг фильтрации
│   ├── finetune_DDPM/
│   │   └── default.yaml            # Конфиг файн-тюнинга DDPM
│   ├── finetune_TDnCNN/
│   │   └── default.yaml            # Конфиг файн-тюнинга TDnCNN
│   ├── gen_noisy_dataset/
│   │   └── default.yaml            # Конфиг генерации зашумленных данных
│   ├── hydra/
│   │   └── hydra_config.yaml       # Системные настройки Hydra
│   ├── preprocessing/
│   │   └── default.yaml            # Конфиг предобработки
│   ├── train_DDPM/
│   │   └── default.yaml            # Конфиг обучения DDPM
│   ├── train_TDnCNN/
│   │   └── default.yaml            # Конфиг обучения TDnCNN
│   └── config.yaml                 # Точка входа конфигурации (Main Config)
├── data/                           # Датасеты (Quadriga datasets)
│   ├── test/
│   └── train/
├── experiments/                    # Логи, чекпоинты и результаты запусков
├── scripts/                        # Исполняемые скрипты
│   ├── evaluation_pipe.py          
│   ├── filter_dataset.py           
│   ├── generate_noisy_dataset.py   
│   ├── main.py                     # Основной скрипт
│   ├── preprocessing_pipe.py       
│   ├── train_DDPM.py               
│   └── train_TDnCNN.py             
└── src/                            
    ├── input_repo/                 # код от H
    ├── datasets.py                 
    ├── DDPM_model.py               
    ├── filters.py                                     
    ├── tools.py                    
    └── Unet_model.py       
```

    


## Пайплайн использования

1. **Подготовка данных:**
   В разделах `data/train` и `data/test` должны быть `.mat` файлы Quadriga соответствующих датасетов (train или test).

2. **Запуск:**
   Запуск осуществляется (если без докера) путем запуска `scripts/main.py` с указанием параметра `task`.

   Пример скрипта:
   ```bash
   python scripts/main.py task='preprocessing'
   ```
   Работает поэтапно. Полный перечень тасков в `main.py` с описанием за что отвечают.
   Для каждого task есть свой файл конфигурации в `configs/`.

4. **Шаг 1: Препроцессинг (Обязательный)**
   Запускаем `preprocessing_pipe.py`. Скрипт прогоняет Quadriga dataset из `data/train` через трансформеры. Обработанный датасет сохраняется в рабочую директорию в `experiments/data` в формате `.h5`.
   Все пути и параметры считываются из `configs/preprocessing/default.yaml`.
   
   Пример скрипта:
   ```bash
   python scripts/main.py task='preprocessing'
   ```

5. **Шаг 2: Генерация шума (Обязательный)**
   Запускаем генерацию зашумленного датасета. Скрипт зашумляет наш `train_dataset` (который был создан на этапе `preprocessing`) шумом SNR [-20, 0] dB (default).
   
   Пример скрипта:
   ```bash
   python scripts/main.py task='gen_noisy_dataset'
   ```

6. **Шаг 3: Обучение (Train / Finetune)**
   Далее можно параллельно запускать с передачей требуемых параметров из конфигов.
   **По умолчанию обучение осуществляется на зашумленном train датасете, который был сформирован на шаге 2.**
   Параметры моделей сохраняются в `experiments/exp_.../models/DDPM_model` (или `TDnCNN_model`).
   
   Примеры скриптов:

   **Обучение DDPM**
   ```bash
   python scripts/main.py task='train_DDPM' train_DDPM.n_epochs=30
   ```

   **Дообучение DDPM с нужной эпохи**
   ```bash
   python scripts/main.py task='finetune_DDPM' finetune_DDPM.checkpoint_name='epoch_002.pth' finetune_DDPM.n_epochs=2
   ```

   **Обучение TDnCNN (денойзер)**
   ```bash
   python scripts/main.py task='train_TDnCNN' train_TDnCNN.n_epochs=50
   ```

   **Дообучение TDnCNN (денойзер)**
   ```bash
   python scripts/main.py task='finetune_TDnCNN' finetune_TDnCNN.checkpoint_name='epoch_002.pth' finetune_TDnCNN.n_epochs=2
   ```

7. **Шаг 4: Фильтрация (Optional)**
   После `train_DDPM` можно запускать фильтрацию датасета. `filter_mode` имеет три варианта: `['top_k', 'QQ_spread', 'random']`.
   В результате в директорию `experiments/exp_.../data` будет сохранен `dataset_filtered.h5`.
   
   Пример скрипта:
   ```bash
   python scripts/main.py task='filter_dataset' filter_dataset.filter_mode='top_k' filter_dataset.save_percent=10
   ```

8. **Шаг 5: Обучение на отфильтрованном датасете**
   Обучаем денойзер TDnCNN на усеченном датасете с явным указанием пути к датасету и `id_label` для новой модели (чтобы не затереть старую).
   
   Пример скрипта:
   ```bash
   python scripts/main.py task='train_TDnCNN' \
   train_TDnCNN.dataset_path="./data/dataset_filtered_top_k_10/filtered_dataset.h5" \
   train_TDnCNN.id_label='filtered_top_k_10' \
   train_TDnCNN.n_epochs=2
   ```

### Конфигурация и пути (Hydra)

*   В `configs/config.yaml` указывается `project.id` (по умолчанию `"exp_001"`). В соответствии с ним в папке `./experiments` создается папка `exp_001`, которая становится рабочей.
*   Все пути в конфигурациях указываются **относительно этой рабочей папки**.

**ВАЖНО:**
*   Помимо `input_dir` для `task='preprocessing'` (по умолчанию `"./data/train"`) и `input_dir` для `task='epochs_evaluation'` (по умолчанию `"./data/test"`). **Эти пути считываются абсолютно** -> можно указать сетевые папки.

### Логирование

*   Весь процесс выполнения каждого `task` можно контролировать в **лог-файле** (`.log`), который автоматически сохраняется в созданной рабочей директории (например, `experiments/exp_001/main.log`).

### Docker

*   Если используем докер, то просто пробрасываем пути к `data/train` и `data/test` от сетевых папок и используем дефолтные конфиги.



