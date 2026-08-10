# Аудит TDnCNN image-адаптации относительно legacy TDCNN и DnCNN

## 1. Краткий вывод

- Итоговый вердикт: **B. Функционально эквивалентная DnCNN/TDCNN-постановка, но не строго идентичная реализация**.

`src/TDnCNN_image.py` сохраняет главный внешний смысл legacy TDCNN как downstream denoiser: на вход подается зашумленный объект, на выходе ожидается clean estimate, обучение идет по MSE/reconstruction loss к clean target. Это подтверждается legacy training loop `x_noisy = x_batch + Noise_batch`, `x_pred = model(x_noisy)`, `loss = (x_pred - x_batch)...` в `archive_legacy/train_TDnCNN.py:107-114` и image loop `x_pred = model(x_noisy)`, `loss = criterion(x_pred, x_clean)` в `src/TDnCNN_image.py:109-114`.

Однако это не строгая идентичная архитектурная постановка. Legacy `DnCNN_3D` является комплексной Conv3d-сетью с `Cplx_cardioid`, circular same padding и direct clean-output semantics (`return out`) в `archive_legacy/TDnCNN.py:16-64`. Image `DnCNN_2D` является вещественной Conv2d-сетью с BatchNorm2d/ReLU и внутренней residual-схемой `noise_pred = self.net(x); return x - noise_pred` в `src/TDnCNN_image.py:13-32`. Поэтому утверждение "та же постановка, только под изображения" корректно только в функциональном смысле noisy-to-clean denoising, но некорректно как утверждение о полной идентичности forward semantics и архитектурного паттерна.

Обе модели можно описывать как DnCNN-inspired / DnCNN-подобные сверточные денойзеры, но с разной строгостью. Image-модель структурно ближе к DnCNN: Conv+ReLU, Conv+BatchNorm+ReLU в середине, Conv в конце, residual subtraction и MSE reconstruction loss (`src/TDnCNN_image.py:13-32`, `src/TDnCNN_image.py:109-114`, `scripts/internal/train_TDnCNN_image.py:185-188`). Legacy-модель не подтверждает strict DnCNN residual learning: она напрямую возвращает clean estimate и обучается на `MSE(model(noisy), clean)`, без явного `noisy - predicted_noise` в forward или training loop (`archive_legacy/TDnCNN.py:59-64`, `archive_legacy/train_TDnCNN.py:110-114`).

Главные совпадения:
- обе модели являются fully convolutional denoisers без pooling/upsampling в самой TDnCNN/DnCNN-сети (`archive_legacy/TDnCNN.py:14-55`, `src/TDnCNN_image.py:13-28`);
- обе сохраняют размер входа/выхода: legacy reshape обратно к `[B, N_ue, N_bs, N_f]`, image padding=1 при kernel=3 возвращает `[B, C, H, W]` (`archive_legacy/TDnCNN.py:59-64`, `src/TDnCNN_image.py:14-32`);
- обе обучаются на clean target после добавления шума к clean input (`archive_legacy/train_TDnCNN.py:107-114`, `src/tdncnn_datasets.py:21-27`, `src/TDnCNN_image.py:109-114`);
- обе используют MSE-like reconstruction objective (`archive_legacy/train_TDnCNN.py:114`, `src/TDnCNN_image.py:114`, `scripts/internal/train_TDnCNN_image.py:185-188`);
- обе работают как downstream validation denoiser в проектной документации (`README.md:202-204`, `docs/PIPELINE.md:406-420`).

Главные расхождения:
- legacy использует complex Conv3d, image использует real Conv2d (`archive_legacy/TDnCNN.py:16-50`, `src/TDnCNN_image.py:14-27`);
- legacy не содержит BatchNorm и ReLU, image содержит BatchNorm2d и ReLU (`archive_legacy/TDnCNN.py:26-41`, `src/TDnCNN_image.py:15-23`);
- legacy forward возвращает выход сети напрямую, image forward вычитает предсказанный шум (`archive_legacy/TDnCNN.py:59-64`, `src/TDnCNN_image.py:30-32`);
- legacy large имеет 15 conv-слоев и 1,466,248 complex parameters, image default имеет 8 conv-слоев и 223,553/225,859 real parameters для 1/3 каналов (`archive_legacy/TDnCNN.py:103-116`, `src/TDnCNN_image.py:10-28`);
- legacy training imports reference absent modules `src.input_repo.*`, отсутствующие в текущем дереве, поэтому полный legacy pipeline не воспроизводится только из найденных файлов (`archive_legacy/train_TDnCNN.py:23-28`).

## 2. Проверенные источники в репозитории

| Файл | Что проверено | Релевантные строки/функции/классы |
|---|---|---|
| `README.md` | Общий scope проекта, dataset configs, TDnCNN downstream validation | `README.md:1-5`, `README.md:21-25`, `README.md:202-245` |
| `docs/PIPELINE.md` | Dataset shapes, active TDnCNN image pipeline, entrypoints | `docs/PIPELINE.md:20-25`, `docs/PIPELINE.md:406-420`, `docs/PIPELINE.md:422-472` |
| `docs/ENCODERS.md` | Проверено, что файл про автоэнкодеры, не про TDnCNN | `docs/ENCODERS.md:1-18`, `docs/ENCODERS.md:28-46` |
| `docs/README.md` | Объем документации и current scope | `docs/README.md:1-14` |
| `archive_legacy/TDnCNN.py` | Legacy model class, forward, complex activation, initialization, factory | `archive_legacy/TDnCNN.py:6-64`, `archive_legacy/TDnCNN.py:67-85`, `archive_legacy/TDnCNN.py:103-116` |
| `archive_legacy/train_TDnCNN.py` | Legacy training/validation objective, noisy construction, clean target | `archive_legacy/train_TDnCNN.py:90-114`, `archive_legacy/train_TDnCNN.py:139-161`, `archive_legacy/train_TDnCNN.py:227-269` |
| `archive_legacy/train.py` | Older compact legacy training example and preprocessing transforms | `archive_legacy/train.py:9-15`, `archive_legacy/train.py:17-31`, `archive_legacy/train.py:98-100` |
| `archive_legacy/scripts/evaluation_pipe.py` | Legacy inference use as denoiser | `archive_legacy/scripts/evaluation_pipe.py:84-116` |
| `src/TDnCNN_image.py` | Image model class, forward, train/eval loss and metrics | `src/TDnCNN_image.py:9-32`, `src/TDnCNN_image.py:99-122`, `src/TDnCNN_image.py:146-156` |
| `src/tdncnn_datasets.py` | Image noisy-clean pair generation, tensor profile, filtering | `src/tdncnn_datasets.py:10-27`, `src/tdncnn_datasets.py:41-54`, `src/tdncnn_datasets.py:76-130` |
| `scripts/internal/train_TDnCNN_image.py` | Image training entry implementation, model construction, MSELoss, optimizer | `scripts/internal/train_TDnCNN_image.py:139-188`, `scripts/internal/train_TDnCNN_image.py:199-228` |
| `scripts/internal/tdncnn_image_runs_config.py` | Dataset-specific in_channels and TDnCNN run configuration | `scripts/internal/tdncnn_image_runs_config.py:68-120` |
| `scripts/internal/run_TDnCNN_image_suite.py` | Public CLI routing for image TDnCNN runs | `scripts/internal/run_TDnCNN_image_suite.py:38-90` |
| `src/dataset_registry.py` | MNIST/CIFAR channel counts and transform profiles | `src/dataset_registry.py:10-29`, `src/dataset_registry.py:89-110` |
| `configs/dataset/mnist.yaml` | MNIST channel/image-size config | `configs/dataset/mnist.yaml:4-13` |
| `configs/dataset/cifar10.yaml` | CIFAR-10 channel/image-size config | `configs/dataset/cifar10.yaml:4-13` |

## 3. Legacy TDCNN: архитектура и постановка

Ключевая legacy model class: `DnCNN_3D` в `archive_legacy/TDnCNN.py:6-64`. Архитектура строится как последовательность Conv3d-слоев: первый Conv3d `channels -> features`, затем `num_of_layers - 2` внутренних Conv3d `features -> features`, затем последний Conv3d `features -> channels` (`archive_legacy/TDnCNN.py:14-55`). Все Conv3d используют `kernel_size=3`, `padding="same"`, `padding_mode="circular"`, `bias=True` (`archive_legacy/TDnCNN.py:16-23`, `archive_legacy/TDnCNN.py:31-39`, `archive_legacy/TDnCNN.py:45-52`).

BatchNorm в legacy `DnCNN_3D` не найден. Вместо ReLU используется комплексная активация `Cplx_cardioid`: она берет phase через `torch.angle(x)` и возвращает `0.5 * (1 + cos(phase)) * x` (`archive_legacy/TDnCNN.py:67-74`). Инициализация принудительно задает комплексные веса и bias: `torch.empty_like(..., dtype=torch.complex64)`, Xavier отдельно для `.real` и `.imag`, bias complex zeros (`archive_legacy/TDnCNN.py:77-85`).

Forward legacy-модели:

```text
input x: [N_batch, N_ue, N_bs, N_f], complex
x_view = x.view(N_batch, N_ue * 2, 4, 8, N_f)
out = tdncnn(x_view)
return out.view(N_batch, N_ue, N_bs, N_f)
```

Это следует из `archive_legacy/TDnCNN.py:59-64`. В forward нет `x - predicted_noise`, нет сложения residual, и нет явного возврата noise/residual. Поэтому по найденному коду legacy-модель возвращает clean estimate напрямую.

Factory `build_DnCNN_3D_model` создает `small` и `large` варианты с `channels=8`, `num_of_layers=15`, `num_features=16` или `64` (`archive_legacy/TDnCNN.py:103-116`). Легкая проверка локальным импортом показала 97,000 complex parameters для `small` и 1,466,248 complex parameters для `large`, то есть 194,000 и 2,932,496 вещественных скаляров соответственно. Dummy forward для `large` подтвердил форму `(2,4,64,96) -> (2,4,64,96)` и `torch.complex64`.

Training objective в основном legacy trainer:

```text
Noise_batch = get_noise(x_batch, SNR_dB=current_snr)
y = x_noisy = x_batch + Noise_batch
x_pred = model(y)
L = sum(|x_pred - x_batch|^2) / batch_size
```

Это прямо записано в `archive_legacy/train_TDnCNN.py:107-114`; validation повторяет ту же схему в `archive_legacy/train_TDnCNN.py:154-161`. Старый компактный `archive_legacy/train.py` делает то же самое для `H_batch`: `H_noisy = H_batch + Noise_batch`, `H_pred = model(H_noisy)`, `mse = (H_pred - H_batch).abs().pow(2).sum() / 10` (`archive_legacy/train.py:21-31`).

Работа с комплексным сигналом подтверждается тремя местами: input приводится к `torch.complex64` перед моделью (`archive_legacy/train_TDnCNN.py:110-112`, `archive_legacy/train_TDnCNN.py:157-159`), веса Conv3d и bias инициализируются как `complex64` (`archive_legacy/TDnCNN.py:77-85`), а активация использует фазу комплексного тензора (`archive_legacy/TDnCNN.py:71-74`). В найденном legacy-коде не подтверждается представление complex signal как real/imag двух вещественных каналов. Напротив, используется комплексный dtype; channel reshape `N_ue * 2` в `archive_legacy/TDnCNN.py:59-62` не сопровождается комментарием, который доказывал бы, что это real/imag split.

Ограничение воспроизводимости: `archive_legacy/train_TDnCNN.py` импортирует `src.input_repo.tools` и `src.input_repo.TDnCNN` (`archive_legacy/train_TDnCNN.py:23-28`), но директория `src/input_repo` в текущем дереве отсутствует. Поэтому полный legacy training pipeline не подтверждается как запускаемый из текущей копии; архитектурная и математическая постановка проверена по сохраненным файлам.

## 4. TDnCNN_image: архитектура и постановка

Ключевая image model class: `DnCNN_2D` в `src/TDnCNN_image.py:9-32`. Конструктор принимает `in_channels: int = 1`, `num_layers: int = 8`, `features: int = 64` (`src/TDnCNN_image.py:9-10`). Первый блок: `Conv2d(in_channels, features, kernel_size=3, padding=1)` и `ReLU(inplace=True)` (`src/TDnCNN_image.py:13-16`). Средние `num_layers - 2` блоков: `Conv2d(features, features, 3, padding=1)`, `BatchNorm2d(features)`, `ReLU(inplace=True)` (`src/TDnCNN_image.py:18-25`). Последний слой: `Conv2d(features, in_channels, kernel_size=3, padding=1)` (`src/TDnCNN_image.py:27-28`).

Forward image-модели:

```text
noise_pred = self.net(x)
return x - noise_pred
```

Это указано в `src/TDnCNN_image.py:30-32`. То есть внешне `model(x_noisy)` возвращает clean estimate, но внутренне сеть `self.net` учится выдавать residual/noise, который вычитается из входа.

Training objective image-модели:

```text
for x_noisy, x_clean in loader:
    x_pred = model(x_noisy)
    loss = criterion(x_pred, x_clean)
```

Это следует из `src/TDnCNN_image.py:109-114`; `criterion = nn.MSELoss()` задается в runner (`scripts/internal/train_TDnCNN_image.py:185-188`). Evaluation использует ту же пару `x_noisy, x_clean`, `x_pred = model(x_noisy)`, `loss = criterion(x_pred, x_clean)` (`src/TDnCNN_image.py:146-151`).

Подготовка noisy-clean пар для изображений находится в `OnlineNoisyDataset`: clean image берется из torchvision dataset, затем `sigma` равномерно выбирается в `[sigma_min, sigma_max]`, `epsilon = torch.randn_like(x_clean)`, `x_noisy = x_clean + sigma * epsilon`, возвращается `(x_noisy.float32, x_clean.float32)` (`src/tdncnn_datasets.py:10-27`). Для TDnCNN используется `transform_profile="tensor"`, а комментарий явно говорит, что TDnCNN работает с clean images в `[0,1]` (`src/tdncnn_datasets.py:41-54`). В `dataset_registry` профиль `tensor` соответствует `transforms.ToTensor()` без Normalize (`src/dataset_registry.py:89-110`).

Канальность задается dataset config и run config: MNIST имеет `channels: 1`, `image_size: 28` (`configs/dataset/mnist.yaml:4-13`), CIFAR-10 имеет `channels: 3`, `image_size: 32` (`configs/dataset/cifar10.yaml:4-13`), а `in_channels` для TDnCNN берется из `DATASET_SPECS[dataset_name]["channels"]` (`scripts/internal/tdncnn_image_runs_config.py:91-105`). Документация подтверждает shapes MNIST `[B,1,28,28]` и CIFAR-10 `[B,3,32,32]` (`docs/PIPELINE.md:20-25`).

Легкая проверка с metric-stub импортом подтвердила shape preservation и параметры: `DnCNN_2D(in_channels=1)` имеет 223,553 real parameters и `(2,1,28,28) -> (2,1,28,28)`; `DnCNN_2D(in_channels=3)` имеет 225,859 real parameters и `(2,3,32,32) -> (2,3,32,32)`. Эти числа также следуют из слоев `src/TDnCNN_image.py:13-28`: 1 first Conv2d, 6 middle Conv2d+BN blocks, 1 final Conv2d.

## 5. Таблица архитектурного сравнения

| Признак | Legacy TDCNN | TDnCNN_image | Совпадает? | Комментарий |
|---|---|---|---|---|
| input shape | `[B, N_ue, N_bs, N_f]`, затем view в `[B, N_ue*2, 4, 8, N_f]` (`archive_legacy/TDnCNN.py:59-62`) | `[B, C, H, W]` для MNIST/CIFAR (`docs/PIPELINE.md:20-25`, `src/tdncnn_datasets.py:19-27`) | Частично | Оба batched tensors, но разные domain dimensions. |
| output shape | Возврат к `[B, N_ue, N_bs, N_f]` (`archive_legacy/TDnCNN.py:62-64`) | `[B, C, H, W]` из-за `padding=1` и `return x - noise_pred` (`src/TDnCNN_image.py:14-32`) | Да по принципу | Оба сохраняют размерность входа. |
| Conv1d/Conv2d/Conv3d | Complex `nn.Conv3d` (`archive_legacy/TDnCNN.py:16-50`, `archive_legacy/TDnCNN.py:77-85`) | Real `nn.Conv2d` (`src/TDnCNN_image.py:14-27`) | Нет | Это ключевая domain adaptation и одновременно архитектурное отличие. |
| Число слоев | Factory: 15 conv layers (`archive_legacy/TDnCNN.py:103-116`) | Default: 8 conv layers (`src/TDnCNN_image.py:10-28`) | Нет | Image default существенно мельче по глубине. |
| Kernel size | `kernel_size=3` для Conv3d (`archive_legacy/TDnCNN.py:10-20`) | `kernel_size=3` для Conv2d (`src/TDnCNN_image.py:14-27`) | Да по размеру ядра | Размер ядра совпадает, размерность ядра нет. |
| Padding | `padding="same"`, `padding_mode="circular"` (`archive_legacy/TDnCNN.py:20-21`, `archive_legacy/TDnCNN.py:35-37`, `archive_legacy/TDnCNN.py:49-50`) | `padding=1`, default zero padding (`src/TDnCNN_image.py:14-27`) | Частично | Оба сохраняют форму, но boundary condition отличается. |
| BatchNorm | Не найден в `DnCNN_3D` (`archive_legacy/TDnCNN.py:14-55`) | `BatchNorm2d` в средних слоях (`src/TDnCNN_image.py:21-23`) | Нет | Image ближе к canonical DnCNN pattern. |
| ReLU/активация | `Cplx_cardioid`, не ReLU (`archive_legacy/TDnCNN.py:26-41`, `archive_legacy/TDnCNN.py:67-74`) | `ReLU(inplace=True)` (`src/TDnCNN_image.py:15`, `src/TDnCNN_image.py:23`) | Нет | Legacy activation complex-specific. |
| residual/noise prediction | Не найдено; forward возвращает `out` (`archive_legacy/TDnCNN.py:59-64`) | `noise_pred = self.net(x)` и `return x - noise_pred` (`src/TDnCNN_image.py:30-32`) | Нет | Главное отличие в DnCNN semantics. |
| clean reconstruction formula | `x_hat = f_theta(y)` (`archive_legacy/train_TDnCNN.py:110-114`) | `x_hat = y - g_theta(y)` (`src/TDnCNN_image.py:30-32`, `src/TDnCNN_image.py:109-114`) | Частично | Внешний output clean estimate совпадает, внутренняя формула нет. |
| loss | `sum(|x_pred - x_clean|^2)/B` (`archive_legacy/train_TDnCNN.py:112-114`) | `MSELoss(x_pred, x_clean)` (`src/TDnCNN_image.py:113-114`, `scripts/internal/train_TDnCNN_image.py:185-188`) | Да по target | Оба reconstruction MSE к clean, но image эквивалентно residual noise MSE для внутреннего `self.net`. |
| target | Clean signal `x_batch` (`archive_legacy/train_TDnCNN.py:107-114`) | Clean image `x_clean` (`src/tdncnn_datasets.py:21-27`, `src/TDnCNN_image.py:109-114`) | Да | Target clean в обоих. |
| parameter count | `small`: 97,000 complex params; `large`: 1,466,248 complex params (`archive_legacy/TDnCNN.py:103-116`) | `C=1`: 223,553 real params; `C=3`: 225,859 real params (`src/TDnCNN_image.py:10-28`) | Нет | Разная емкость и dtype. |
| сохранение размерности | Да (`archive_legacy/TDnCNN.py:59-64`) | Да (`src/TDnCNN_image.py:14-32`) | Да | Подтверждено также dummy forward. |
| pooling/upsampling | Не найдено в `DnCNN_3D` (`archive_legacy/TDnCNN.py:14-55`) | Не найдено в `DnCNN_2D` (`src/TDnCNN_image.py:13-28`) | Да | Fully convolutional без down/up sampling. |
| соответствие DnCNN | Частично: сверточный denoiser + MSE clean, но нет residual noise prediction и BN/ReLU pattern | Да/почти: Conv-BN-ReLU pattern + residual subtraction + MSE reconstruction | Частично | Legacy является DnCNN-inspired, image ближе к DnCNN. |

## 6. Сравнение математической постановки

Обозначения:

- `x` - clean signal/image;
- `n` - synthetic noise;
- `y = x + n` - noisy input;
- `f_theta` - вся legacy-модель;
- `g_theta` - внутренняя `self.net` image-модели;
- `F_theta(y) = y - g_theta(y)` - внешний output image-модели.

Legacy mapping по найденному коду:

```text
y = x + n
x_hat = f_theta(y)
L_legacy = ||x_hat - x||^2
```

Evidence: `x_noisy = x_batch + Noise_batch`, `x_pred = model(x_noisy)`, `loss = (x_pred - x_batch).abs().pow(2).sum()/current_bs` в `archive_legacy/train_TDnCNN.py:107-114`. Forward возвращает `out`, а не residual operation (`archive_legacy/TDnCNN.py:59-64`).

Image mapping:

```text
y = x + n
r_hat = g_theta(y)
x_hat = F_theta(y) = y - r_hat
L_image = ||x_hat - x||^2 = ||(y - x) - r_hat||^2
```

Evidence: noisy pair construction в `src/tdncnn_datasets.py:21-27`, residual forward в `src/TDnCNN_image.py:30-32`, MSE to clean в `src/TDnCNN_image.py:109-114` и `scripts/internal/train_TDnCNN_image.py:185-188`.

Формулы совпадают на уровне внешней supervised denoising задачи `noisy -> clean` с reconstruction MSE. Они не совпадают на уровне strict residual DnCNN semantics: legacy не показывает `r_hat = model(y)` и `x_hat = y - r_hat`; image показывает именно эту схему внутри `DnCNN_2D.forward`.

Sign residual: image использует вычитание (`return x - noise_pred`, `src/TDnCNN_image.py:30-32`). В legacy знак residual не определен, потому что residual operation отсутствует (`archive_legacy/TDnCNN.py:59-64`).

Нормализация: для image TDnCNN датасет явно использует clean images в `[0,1]` и `ToTensor()` (`src/tdncnn_datasets.py:47-54`, `src/dataset_registry.py:100-101`). Общие dataset configs описывают Normalize mean/std 0.5 (`configs/dataset/mnist.yaml:12-13`, `configs/dataset/cifar10.yaml:12-13`) и документация говорит о `[-1,1]` для общей dataset layer (`docs/PIPELINE.md:20-25`), но TDnCNN image path в коде выбирает `transform_profile="tensor"`, поэтому для данного аудита фактическая TDnCNN-нормализация берется из кода.

## 7. Проверка наследования от DnCNN

| Признак DnCNN | Есть в legacy? | Есть в TDnCNN_image? | Доказательство из кода | Комментарий |
|---|---|---|---|---|
| Fully convolutional denoiser | Да | Да | Legacy только Conv3d+activation в `archive_legacy/TDnCNN.py:14-55`; image Conv2d+BN+ReLU в `src/TDnCNN_image.py:13-28` | Оба без Linear/pooling в denoiser body. |
| Одинаковый размер входа и выхода | Да | Да | `view(...)->tdncnn->view(...)` в `archive_legacy/TDnCNN.py:59-64`; padding=1 и subtraction в `src/TDnCNN_image.py:14-32` | Подтверждено dummy forward. |
| Residual learning / prediction of noise | Нет по найденному коду | Да внутри `forward` | Legacy `return out` в `archive_legacy/TDnCNN.py:59-64`; image `noise_pred` и `return x - noise_pred` в `src/TDnCNN_image.py:30-32` | Ключевое различие. |
| Conv + ReLU first layer | Нет | Да | Legacy `Conv3d` + `Cplx_cardioid` (`archive_legacy/TDnCNN.py:16-27`); image `Conv2d` + `ReLU` (`src/TDnCNN_image.py:13-16`) | Legacy activation complex-specific. |
| Conv + BatchNorm + ReLU middle layers | Нет | Да | Legacy middle Conv3d + `Cplx_cardioid` (`archive_legacy/TDnCNN.py:29-41`); image middle Conv2d + BatchNorm2d + ReLU (`src/TDnCNN_image.py:18-25`) | Image соответствует DnCNN pattern. |
| Final Conv without activation | Да | Да | Legacy final Conv3d appended after loop (`archive_legacy/TDnCNN.py:43-55`); image final Conv2d (`src/TDnCNN_image.py:27-28`) | Совпадает по принципу. |
| MSE or analogous loss | Да | Да | Legacy squared magnitude loss (`archive_legacy/train_TDnCNN.py:112-114`); image `nn.MSELoss` (`scripts/internal/train_TDnCNN_image.py:185-188`) | Legacy loss complex magnitude. |
| Loss target is noise/residual | Нет | Эквивалентно, но не явно | Legacy target clean (`archive_legacy/train_TDnCNN.py:112-114`); image target clean после residual forward (`src/TDnCNN_image.py:109-114`) | Image reconstruction MSE эквивалентен MSE на `noise_pred` к `y-x`. |
| No pooling/upsampling | Да | Да | `archive_legacy/TDnCNN.py:14-55`, `src/TDnCNN_image.py:13-28` | В самих denoisers pooling/upsampling не найдено. |
| Downstream denoiser use | Да | Да | Legacy evaluation `denoised_batch = TDnCNN_model(noisy_batch)` (`archive_legacy/scripts/evaluation_pipe.py:115-116`); docs list image TDnCNN validation (`docs/PIPELINE.md:406-420`) | Назначение совпадает. |

Вывод: image-модель является DnCNN-подобной residual image denoising реализацией. Legacy-модель является сверточным комплексным denoiser, вдохновленным DnCNN названием/формой, но strict DnCNN residual learning по найденному коду не подтверждается.

## 8. Что является корректной адаптацией под изображения

Переход `Conv3d -> Conv2d` может быть корректной размерностной адаптацией, если меняется только domain geometry: вместо комплексного/многомерного канального тензора `[B, N_ue, N_bs, N_f]` используются изображения `[B, C, H, W]`, а сеть остается fully convolutional, shape-preserving и обучается denoising mapping. В image-коде это выполнено: Conv2d принимает `in_channels`, сохраняет spatial size через `padding=1`, а данные приходят как `[B,C,H,W]` из torchvision (`src/TDnCNN_image.py:10-32`, `src/tdncnn_datasets.py:19-27`, `docs/PIPELINE.md:20-25`).

Замена комплексного сигнала на изображение не меняет постановку denoising, если сохраняются условия: noisy input строится как clean + noise, target остается clean, output модели трактуется как clean estimate, loss остается MSE/reconstruction loss или строго эквивалентный residual MSE. Эти условия выполнены внешне для обеих веток (`archive_legacy/train_TDnCNN.py:107-114`, `src/tdncnn_datasets.py:21-27`, `src/TDnCNN_image.py:109-114`).

Чтобы утверждение "та же постановка, только под изображения" было строго корректным, должны совпадать не только noisy-to-clean target, но и forward semantics. В частности, если legacy должен считаться DnCNN-style residual denoiser, в коде legacy должно быть подтверждено одно из двух: `r_hat = model(y); x_hat = y - r_hat` в training/evaluation или `forward` должен сам возвращать `y - predicted_noise`. В найденном legacy-коде этого нет (`archive_legacy/TDnCNN.py:59-64`, `archive_legacy/train_TDnCNN.py:110-114`).

## 9. Найденные проблемы, риски или неоднозначности

| Severity | Evidence | Impact | Recommendation |
|---|---|---|---|
| major | Legacy forward возвращает `out`, image forward возвращает `x - noise_pred` (`archive_legacy/TDnCNN.py:59-64`, `src/TDnCNN_image.py:30-32`) | Нельзя утверждать strict identity forward semantics или strict legacy residual DnCNN. | В НИР писать "функционально сохраняет noisy-to-clean denoising objective", а не "полностью идентичная residual formulation". |
| major | Legacy loss target clean: `(x_pred - x_batch)` (`archive_legacy/train_TDnCNN.py:112-114`); image loss target clean: `criterion(x_pred, x_clean)` (`src/TDnCNN_image.py:109-114`) | Обе постановки реконструкционные, но legacy не обучает модель напрямую на noise target. | Для DnCNN-сравнения явно указать, что image residual loss эквивалентен noise-MSE, а legacy является direct clean estimator. |
| major | Legacy complex Conv3d + `Cplx_cardioid`, image real Conv2d + BN/ReLU (`archive_legacy/TDnCNN.py:16-50`, `archive_legacy/TDnCNN.py:67-74`, `src/TDnCNN_image.py:14-27`) | Архитектурная идентичность не подтверждается; это перенос идеи, а не layer-by-layer port. | Формулировать как "адаптация архитектурной идеи под image domain". |
| major | `archive_legacy/train_TDnCNN.py` импортирует `src.input_repo.*`, отсутствующий в текущем дереве (`archive_legacy/train_TDnCNN.py:23-28`) | Полная воспроизводимость legacy pipeline из текущего checkout не подтверждается. | В НИР ссылаться на доступные legacy source snapshots; если нужно воспроизводить, вернуть `src/input_repo` или обновить imports. |
| minor | Legacy uses circular same padding, image uses zero padding=1 (`archive_legacy/TDnCNN.py:20-21`, `src/TDnCNN_image.py:14-27`) | Boundary behavior отличается, может влиять на denoising. | Указать как domain-specific отличие. |
| minor | Legacy factory uses 15 layers, image default 8 layers (`archive_legacy/TDnCNN.py:103-116`, `src/TDnCNN_image.py:10-28`) | Capacity/depth отличаются; strict architecture identity невозможна. | Если нужна более сильная аналогия, синхронизировать `num_layers=15` в image experiments или описать отличие. |
| minor | Legacy has complex Xavier init; image uses PyTorch defaults (`archive_legacy/TDnCNN.py:77-85`, `src/TDnCNN_image.py:13-28`) | Различие initialization влияет на training dynamics. | Отразить как implementation detail, не как изменение denoising objective. |
| note | Общая документация говорит о Normalize to `[-1,1]`, TDnCNN dataset code использует `[0,1]` tensor profile (`docs/PIPELINE.md:20-25`, `src/tdncnn_datasets.py:47-54`, `src/dataset_registry.py:100-101`) | Нормализация TDnCNN image отличается от общей dataset layer. | В описании эксперимента TDnCNN указывать фактический `[0,1]` режим. |
| note | `requirements.txt` содержит `torch-fidelity`, но не содержит `lpips` и `torchmetrics`, хотя `src/TDnCNN_image.py` импортирует их (`requirements.txt:1-13`, `src/TDnCNN_image.py:1-6`) | Импорт полного `src.TDnCNN_image` может зависеть от окружения; это не меняет модельный код, но влияет на запускаемость метрик. | Зафиксировать зависимости для воспроизводимости или перенести metric imports ближе к месту использования. |

## 10. Рекомендованная формулировка для НИР

Вариант 1 - осторожный научный:

> В работе используется адаптация TDCNN/DnCNN-подобного сверточного денойзера к изображениям. Как и legacy TDCNN лаборатории, модель обучается как supervised noisy-to-clean denoiser: к clean объекту добавляется шум, сеть получает noisy input, а качество оптимизируется MSE-реконструкцией clean target. При переносе на изображения комплексные Conv3d-блоки legacy заменены на вещественные Conv2d-блоки, а image-версия реализует DnCNN-подобную residual-схему через вычитание предсказанного шума из входа.

Вариант 2 - сильный, но с корректной оговоркой:

> Предложенная image-версия сохраняет функциональную постановку legacy TDCNN как downstream denoiser: входом является зашумленный объект, выходом - оценка clean объекта, а обучение проводится по reconstruction MSE. При этом реализация не является послойно идентичной legacy-модели: вместо комплексных Conv3d и прямого clean-output forward используется вещественная Conv2d DnCNN-подобная residual-схема `x_hat = y - f_theta(y)`.

## 11. Итоговый ответ консультантке

Были проверены README, docs, legacy model/training snapshots и текущая image-реализация `src/TDnCNN_image.py`. Постановка совпадает в функциональном смысле: обе ветки решают supervised denoising `noisy -> clean` и используют MSE-like reconstruction loss к clean target. Из legacy перенесены fully convolutional shape-preserving denoiser, synthetic noisy input, clean target и downstream validation role. Из-за перехода к изображениям комплексные Conv3d и complex cardioid заменены на вещественные Conv2d, BatchNorm2d и ReLU, а вход стал `[B,C,H,W]` для MNIST/CIFAR. Image-версия является более строгим продолжателем DnCNN residual idea, потому что внутри forward предсказывает шум и вычитает его из входа. Legacy-модель по найденному коду является DnCNN-inspired denoiser, но strict residual DnCNN formulation для нее не подтверждается. Поэтому корректный вывод: это функционально эквивалентная адаптация denoising-постановки, но не строго идентичная реализация legacy TDCNN.
