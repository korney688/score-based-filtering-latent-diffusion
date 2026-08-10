# ImageNet-100 Image-Space Variable-AWGN Audit

Date: 2026-08-09

Scope: adapt the checked CIFAR-10 variable-AWGN image pipeline to a separate ImageNet-100 image-space branch. Full ImageNet-100 is not available locally and was not downloaded.

| Компонент | Текущий ImageNet путь | CIFAR reference | Решение для новой ветки |
|---|---|---|---|
| Dataset config | `configs/dataset/imagenet100.yaml` | `configs/dataset/cifar10.yaml` | New experiment config: `configs/experiments/imagenet100_image_space_variable_awgn_v1.yaml`; ImageFolder root is launcher-overridden. |
| Dataset class | `src/dataset_registry.py::build_torchvision_split`, ImageFolder | `src.dataset_registry.build_torchvision_split(..., transform_profile="tensor")` | Reuse deterministic Resize + CenterCrop + ToTensor for ImageNet; no stochastic crop/flip in corpus path. |
| CIFAR corpus corruption | Not present for ImageNet variable-AWGN | `src/image_lab_replication_datasets.py::materialize_variable_awgn_corpus` and `src/evaluation/cifar10_score_degradation_validation.py::materialize_corpus` | Implemented on-demand ImageNet corpus in `src/imagenet100_image_space_variable_awgn.py` with the same stable SHA256 seed idea and all required manifest fields. |
| CIFAR DDPM score validation | Old ImageNet uses `scripts/evaluate_latent_ddpm_score.py` | `src/evaluation/cifar10_score_degradation_validation.py` | New `score_validate` stage scans ImageNet checkpoints/timesteps in image space and writes `03_score_validation`. |
| CIFAR selected filtering | Old ImageNet uses `scripts/main.py task=filter_dataset` | `scripts/run_cifar_selected_filtering.py` | New `filter` stage reads only `04_selected_score_config/selected_score_config.yaml`; no independent checkpoint/timestep CLI. |
| Quantile-spread | Old ImageNet filtering has separate topk/quantile folders | `src/evaluation/image_lab_replication_analysis.py::select_quantile_spread_exact` | Imported directly, with CIFAR rounding `int(round(ratio * n_full))`. |
| Downstream TDnCNN | `scripts/train_tdncnn.py`, `scripts/internal/train_TDnCNN_image.py` | `scripts/run_cifar_full50k_pipeline.py::train_variant` | New max-step TDnCNN stage uses one shared initial state and cycles filtered subsets to equal optimizer steps. |
| Internal evaluation | `scripts/internal/train_TDnCNN_image.py` validates during training | CIFAR full50k evaluation in `scripts/run_cifar_full50k_pipeline.py` | New `evaluate` stage creates one fixed eval-noise cache and evaluates all variants against clean targets. |
| External benchmarks | `scripts/evaluate_external.py` | CIFAR external pattern absent; existing ImageNet evaluator supports Kodak24/CBSD68/Urban100 | New `external_eval` reuses helper functions from `scripts/evaluate_external.py` and evaluates Full/Top-K/Quantile/Random with sigma 25/255 seed 42. |
| Old ImageNet launcher | `scripts/run_imagenet100_full.sh` | N/A | Preserved. It runs encoder training, encoder validation, latent DDPM baseline/induced, latent score validation, old filtering, TDnCNN. New launcher bypasses it. |
| Noisy old ImageNet launcher | `scripts/run_imagenet100_full_noisy.sh` | N/A | Preserved. It is Poisson-Gaussian + encoder/latent and is not used by the new branch. |
| DGX smoke launcher | `scripts/run_imagenet100_dgx_smoke.sh` | N/A | Used for remote path conventions only: `/workspace`, `/workspace/data/imagenet100`. |
| New launcher | N/A | CIFAR script stage organization | Added `scripts/run_imagenet100_image_space_variable_awgn.sh` with stage selection, resume, preflight, dry-run, markers. |

## Active CIFAR Reference

The active CIFAR variable-AWGN work is in:

- `src/image_lab_replication_datasets.py`: stable seed, corpus metrics, materialized corpus, task/eval noise datasets.
- `src/evaluation/cifar10_score_degradation_validation.py`: score validation, ladder construction, timestep convention, MC scoring, paired Spearman, ordering, bootstrap, stability.
- `scripts/run_cifar_selected_filtering.py`: selected e18/t100 filtering artifacts, score formula `sum(flatten(eps_pred)**2)`, Top-K minimum score, random subset.
- `scripts/run_cifar_full50k_pipeline.py`: full 50k selected pipeline, equal-step TDnCNN downstream, fixed eval cache.
- `src/evaluation/image_lab_replication_analysis.py::select_quantile_spread_exact`: Quantile-spread algorithm.

CIFAR values epoch 18, report timestep 100, code index 99, and MC K=16 are treated as reference context only. They are not ImageNet defaults.

## Encoder/Latent Dependencies Bypassed

The following current ImageNet paths lead into the old encoder/latent graph:

- `configs/dataset/imagenet100.yaml`: `encoder`, `encoder_validation`, `latent_ddpm`, `latent_ddpm_validation`.
- `src/DDPM_model.py`: frozen autoencoder, latent dimension, baseline/induced latent noise.
- `scripts/train_encoder.py`, `scripts/evaluate_encoder.py`.
- `scripts/main.py task=train_latent_DDPM`.
- `scripts/evaluate_latent_ddpm_score.py`.
- `scripts/run_imagenet100_full.sh`, `scripts/run_imagenet100_dgx_smoke.sh`, `scripts/run_imagenet100_full_noisy.sh`.

The new call graph uses `src/imagenet100_image_space_variable_awgn.py::ImageSpaceDDPM` and never requests encoder config, encoder checkpoint, latent dimension, decoder, reconstruction stage, or induced latent noise.

## Resolved Technical Facts

- Image range: clean and corpus tensors are `[0,1]`; DDPM receives `x * 2 - 1`; metric computation stays in `[0,1]`.
- Normalization location: only inside DDPM dataset/scoring path via `normalize_for_ddpm`; downstream and evaluation use `[0,1]`.
- Image size: ImageNet-100 config uses 64x64 RGB.
- `original_index`: split-local ImageFolder index; synthetic smoke also uses split-local index.
- Class names: `ImageFolder.classes[class_id]`; smoke uses `synset_###`.
- Transforms: Resize, CenterCrop, ToTensor; no random crop/flip/color augmentation in corpus branch.
- Local ImageNet count: not available locally. Launcher preflight counts remote train/val samples and requires 100 train/val class directories unless `allow_count_mismatch` is explicitly enabled.
- Checkpoint format: `.pt` dict with `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, optional `scaler_state_dict`, `epoch`, `global_step`, `config_hash`, `prediction_type`.
- Diffusion steps: config default `T=1000`.
- Beta schedule: linear `1e-4 -> 0.02`.
- Timestep convention: report timestep is one-based; code timestep index is `report - 1` for `T=1000`, relative projection otherwise.
- DDPM objective: epsilon prediction MSE.
- Score formula: `mean_K sum(flatten(eps_pred(x_t,t)) ** 2)`, per sample.
- Downstream target/input: `x_target = x_corpus`; `x_input = clip(x_corpus + task_noise, 0, 1)` with separate seed namespace.
- Evaluation noise: one cache shared by all variants; default sigma `25/255`, seed 42.
- Remote paths found in working shell launchers: `PROJECT_ROOT=/workspace`, `IMAGENET100_ROOT=/workspace/data/imagenet100`, `DATA_ROOT=/workspace/data`, external benchmark default `data/external_benchmarks`.

## New Experiment Root

`experiments/imagenet100_image_space_variable_awgn_v1/${RUN_ID}/`

The new launcher defaults to `${OUTPUT_BASE}/imagenet100_image_space_variable_awgn_v1/${RUN_ID}` with `OUTPUT_BASE=/workspace/experiments`.

## Production Workload From Config

- Expected train samples: 130000.
- Expected val samples: 5000.
- Score-validation ladder: 800 images total, 8 per class, 7 PSNR levels.
- Validation corpus: 3000 images total, 30 per class.
- DDPM epochs: 40.
- Checkpoint interval: 1 epoch.
- Coarse checkpoint epochs: 1 through 40.
- Coarse checkpoint count: 40.
- Final candidate count: 3.
- Timestep grid: `[1, 2, 5, 10, 20, 50, 100, 200, 500]`.
- Coarse MC K: 4.
- Final MC K: 16.
- Downstream max optimizer steps: 50000.
- Batch sizes: DDPM 64, score validation 64, filtering 64, downstream 32, evaluation 32.

DDPM forward-evaluation workload requested by protocol:

- Coarse: `800 * 7 * 9 * 4 * 40 = 8,064,000`.
- Final ladder: `800 * 7 * 16 * 3 = 268,800`.
- Final validation corpus: `3000 * 16 * 3 = 144,000`.
- Requested total: `8,476,800`.

The current implementation also computes validation-corpus metrics during coarse scan:

- Additional coarse validation corpus: `3000 * 4 * 9 * 40 = 4,320,000`.
- Implementation total including this metric pass: `12,796,800`.

## Class-Coverage Methodology

Score selection no longer uses the old flat `0.80` validation coverage threshold.

- If validation corpus has at least 3000 images, candidate gate requires validation Top-K class coverage >= 0.95.
- If validation corpus has fewer than 3000 images, candidate gate uses Random-relative coverage and class-distribution JS divergence instead of an arbitrary hard threshold.
- Full-train filtering has a separate hard check: production Top-K 10% must contain all 100 classes.
- Filtering artifacts include min, median, and max selected samples per class, per-class selection rates, Top-K versus Random 10% coverage, and class-distribution JS divergence.

Real-data smoke is explicitly non-scientific and may use relaxed score selection to test wiring after only 2-3 DDPM optimizer steps. Production filtering remains blocked unless score selection passes the scientific gate.
