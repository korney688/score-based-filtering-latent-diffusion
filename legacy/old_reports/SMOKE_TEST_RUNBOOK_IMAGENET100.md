# ImageNet-100 Pipeline Smoke Test Runbook

This runbook is based on the current repository entrypoints and configs. It does
not add code, configs, Dockerfiles, or new entrypoints.

Run all commands manually from the repository root. This runbook is intended to
prove local pipeline operability before building a Docker image, not to validate
scientific quality.

Commands below use `python`. If the project virtual environment is not activated
in PowerShell, use `.\.venv\Scripts\python.exe` instead of `python`.

Active benchmark:

```text
imagenet100
```

Smoke dataset:

```text
data/smoke_imagenet
```

The smoke dataset is generated from CIFAR-10 and stored as ImageFolder data with
the spatial size expected by `configs/dataset/imagenet100.yaml`. The pipeline is
still executed with `dataset=imagenet100`; the dataset root is redirected through
`IMAGENET100_ROOT`.

Pipeline covered:

```text
Encoder Training
-> Encoder Validation
-> Latent DDPM Baseline
-> Latent DDPM Induced
-> Score Validation
-> Score-Based Filtering
-> TDnCNN Downstream Validation
-> Research Plots
```

Audit notes:

- `configs/dataset/imagenet100.yaml` exists.
- `scripts/train_encoder.py` supports ImageNet-100 only through
  `noise-consistency`.
- `train_latent_DDPM.max_samples` is not present in the current DDPM config.
  DDPM training can be reduced to `n_epochs=1` and small `batch_size`, but a
  true train-subset DDPM smoke is not currently exposed by CLI/config.
- TDnCNN ImageNet-100 has a dedicated `topk_10_smoke` run.
- This runbook does not require the full ImageNet-100 dataset. It uses
  `data/smoke_imagenet` only to prove that code paths and artifact routing work.

## 1. Smoke Test Philosophy

A smoke test is not intended to produce a scientific result. Its purpose is to
verify that the ImageNet-100 pipeline can pass through every stage with the
current code, configs, artifact paths, and data layout.

For this runbook, ImageNet-100 data is substituted with a tiny CIFAR-10-derived
ImageFolder dataset at `data/smoke_imagenet`. This is intentional: the goal is
to prove pipeline wiring before Docker image assembly, not to test dataset
quality or benchmark accuracy.

All stages should be run on the smallest volume that the existing entrypoint
supports. Each command is launched manually by the operator in the same
PowerShell session after setting `IMAGENET100_ROOT`. After every stage, the
operator checks that the expected artifacts exist and that logs contain no
Python traceback.

Smoke artifacts are disposable. If a previous valuable run exists under the same
path, move it aside before running these commands.

## 2. Pre-flight Checks

Compile check:

```powershell
python -m compileall -q scripts src
```

Success criteria:

- the command exits with code 0;
- no syntax errors are reported.

Check that the ImageNet-100 dataset config exists:

PowerShell:

```powershell
Test-Path configs/dataset/imagenet100.yaml -PathType Leaf
```

Prepare the local smoke ImageFolder dataset from CIFAR-10:

PowerShell:

```powershell
python scripts/prepare_smoke_imagenet_from_cifar10.py --overwrite
```

Expected successful output:

```text
Smoke ImageFolder dataset created:
  output_dir: .../data/smoke_imagenet
  image_size: 64x64
  train classes: 10
  val classes: 10
  train images: 1000
  val images: 200
```

Set the ImageNet-100 dataset root for the current PowerShell session:

```powershell
$env:IMAGENET100_ROOT = (Resolve-Path data/smoke_imagenet).Path
Write-Host $env:IMAGENET100_ROOT
```

All commands below must be launched from the same PowerShell session. The
project dataset registry checks `IMAGENET100_ROOT` before falling back to
`data/imagenet100`.

Check expected ImageFolder layout:

PowerShell:

```powershell
Test-Path data/smoke_imagenet/train -PathType Container
Test-Path data/smoke_imagenet/val -PathType Container
```

Expected structure:

```text
data/smoke_imagenet/
|-- train/
`-- val/
```

Each split must contain class directories:

PowerShell:

```powershell
Get-ChildItem data/smoke_imagenet/train -Directory | Select-Object -First 10 -ExpandProperty Name
Get-ChildItem data/smoke_imagenet/val -Directory | Select-Object -First 10 -ExpandProperty Name
```

Class and image count checks:

PowerShell:

```powershell
(Get-ChildItem data/smoke_imagenet/train -Directory | Measure-Object).Count
(Get-ChildItem data/smoke_imagenet/val -Directory | Measure-Object).Count
(Get-ChildItem data/smoke_imagenet/train -Recurse -File | Measure-Object).Count
(Get-ChildItem data/smoke_imagenet/val -Recurse -File | Measure-Object).Count
```

Expected counts:

```text
train classes: 10
val classes: 10
train images: 1000
val images: 200
```

## 3. Encoder Smoke

Entrypoint audited:

```text
scripts/train_encoder.py
scripts/internal/train_autoencoder_noise_consistency_mnist.py
configs/dataset/imagenet100.yaml
```

Current CLI supports:

- `--dataset imagenet100`
- `--variant large`
- `--latent-dim 256`
- `--epochs`
- `--fast_dev_run`

Current CLI does not support an encoder `batch_size` override. The batch size is
fixed in the implementation at `128`. `--fast_dev_run` limits the train split to
up to 10,000 samples.

Command:

```bash
python scripts/train_encoder.py noise-consistency --dataset imagenet100 --variant large --latent-dim 256 --epochs 1 --fast_dev_run
```

Expected artifacts:

```text
checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/
|-- autoencoder_checkpoint.pt
`-- E.pt

outputs/imagenet100/autoencoders/noise_consistency_large_latent256/
|-- metrics.json
|-- loss_curve.png
`-- reconstruction_grid.png
```

Success criteria:

- training finishes after 1 epoch;
- `autoencoder_checkpoint.pt` exists;
- `E.pt` exists;
- `outputs/imagenet100/autoencoders/noise_consistency_large_latent256/metrics.json`
  exists;
- stdout/stderr and logs contain no traceback.

## 4. Encoder Validation Smoke

Entrypoint audited:

```text
scripts/evaluate_encoder.py
src/evaluation/encoder_validation.py
```

Command:

```bash
python scripts/evaluate_encoder.py compare-encoders --dataset imagenet100 --num-samples 8 --batch-size 4
```

Expected artifacts:

```text
experiments/imagenet100/exp_002_encoder_validation/
|-- summary.json
|-- metrics/
|   |-- reconstruction_summary.csv
|   |-- reconstruction_summary.md
|   |-- geometry_summary.csv
|   `-- geometry_summary.md
|-- reconstructions/noise_consistency/
|   |-- metrics.json
|   `-- reconstruction_grid.png
|-- latent_geometry/noise_consistency/
|   |-- data.csv
|   |-- metrics.json
|   |-- sigma_vs_score_latent_scatter.png
|   `-- sigma_bins_mean_score_latent.png
`-- report/encoder_validation_report.md
```

Success criteria:

- `summary.json` exists;
- reconstruction and geometry summaries exist under `metrics/`;
- per-encoder `metrics.json` files exist;
- metrics contain non-empty rows for `noise_consistency`;
- log contains no traceback.

## 5. Latent DDPM Baseline Smoke

Entrypoint audited:

```text
scripts/main.py
scripts/train_ddpm.py
configs/train_latent_DDPM/default.yaml
configs/dataset/imagenet100.yaml
```

Current DDPM training config supports `n_epochs`, `batch_size`,
`num_workers`, `id_label`, `latent_noise_mode`, `score_stats`, and DDPM model
shape overrides. It does not support a train-subset override.

The command below uses a separate smoke output directory via the supported
`id_label` field:

```bash
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=baseline train_latent_DDPM.id_label=_smoke train_latent_DDPM.n_epochs=1 train_latent_DDPM.batch_size=2 train_latent_DDPM.num_workers=0 train_latent_DDPM.patience=1 train_latent_DDPM.score_stats.enabled=true train_latent_DDPM.score_stats.max_score_stat_samples=16 train_latent_DDPM.score_stats.score_stat_every_n_epochs=1
```

Expected artifacts:

```text
checkpoints/imagenet100/ddpm/latent_ddpm_baseline_ae_noise_consistency_imagenet100_smoke/
|-- main.log
|-- DDPM_metrics.csv
|-- score_training_dynamics.csv
|-- epoch_001.pth
|-- best_model.pth
`-- ddpm_loss.png
```

Success criteria:

- 1 epoch completes;
- `epoch_001.pth` exists;
- `best_model.pth` exists;
- `DDPM_metrics.csv` contains one row;
- `score_training_dynamics.csv` exists;
- log contains no traceback.

## 6. Latent DDPM Induced Smoke

Entrypoint audited:

```text
scripts/main.py
scripts/train_ddpm.py
configs/train_latent_DDPM/default.yaml
configs/dataset/imagenet100.yaml
```

Command:

```bash
python scripts/main.py dataset=imagenet100 task=train_latent_DDPM train_latent_DDPM.latent_noise_mode=induced train_latent_DDPM.id_label=_smoke train_latent_DDPM.n_epochs=1 train_latent_DDPM.batch_size=2 train_latent_DDPM.num_workers=0 train_latent_DDPM.patience=1 train_latent_DDPM.score_stats.enabled=true train_latent_DDPM.score_stats.max_score_stat_samples=16 train_latent_DDPM.score_stats.score_stat_every_n_epochs=1
```

Expected artifacts:

```text
checkpoints/imagenet100/ddpm/latent_ddpm_induced_ae_noise_consistency_imagenet100_smoke/
|-- main.log
|-- DDPM_metrics.csv
|-- score_training_dynamics.csv
|-- epoch_001.pth
|-- best_model.pth
`-- ddpm_loss.png
```

Success criteria:

- 1 epoch completes;
- `epoch_001.pth` exists;
- `best_model.pth` exists;
- `DDPM_metrics.csv` contains one row;
- `score_training_dynamics.csv` exists;
- log contains no traceback.

## 7. Score Validation Smoke

Entrypoint audited:

```text
scripts/evaluate_latent_ddpm_score.py
src/evaluation/latent_ddpm_score_validation.py
```

Command:

```bash
python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100 --num-samples 8 --batch-size 4 --modes baseline induced --baseline-run-dir checkpoints/imagenet100/ddpm/latent_ddpm_baseline_ae_noise_consistency_imagenet100_smoke --induced-run-dir checkpoints/imagenet100/ddpm/latent_ddpm_induced_ae_noise_consistency_imagenet100_smoke
```

Expected artifacts:

```text
experiments/imagenet100/exp_003_latent_ddpm_validation/
|-- summary.json
|-- metrics/
|   |-- baseline_metrics.json
|   |-- induced_metrics.json
|   |-- summary.csv
|   |-- summary.md
|   |-- distribution_diagnostics.json
|   `-- comparison_metrics.csv
|-- score_validation/
|   |-- baseline/score_samples.csv
|   `-- induced/score_samples.csv
|-- score_distributions/
|-- noise_prediction/
|-- covariance/
`-- report/latent_ddpm_validation_report.md
```

Success criteria:

- `experiments/imagenet100/exp_003_latent_ddpm_validation/summary.json`
  exists;
- both `baseline_metrics.json` and `induced_metrics.json` exist;
- each mode has `score_validation/<mode>/score_samples.csv`;
- `metrics/summary.csv` contains both modes;
- log contains no traceback.

## 8. Filtering Smoke

Entrypoint audited:

```text
scripts/main.py
scripts/filter_dataset.py
configs/filter_dataset/default.yaml
src/filters.py
```

This stage must generate the default top-k 10 percent output because the
ImageNet-100 TDnCNN `topk_10_smoke` run reads:

```text
experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy
```

Command:

```bash
python scripts/main.py dataset=imagenet100 task=filter_dataset filter_dataset.ddpm_branch=induced dataset.latent_ddpm_validation.runs.induced.checkpoint_run=latent_ddpm_induced_ae_noise_consistency_imagenet100_smoke 'filter_dataset.filter_modes=[top_k]' 'filter_dataset.filter_percentages=[10]' filter_dataset.max_samples=16 filter_dataset.batch_size=4 filter_dataset.grid_n_images=8 filter_dataset.noisy_grid_n_images=4 filter_dataset.overwrite=true
```

Expected artifacts:

```text
experiments/imagenet100/exp_005_filtering/
|-- summary.json
|-- metrics/
|   |-- filtering_summary.csv
|   `-- filtering_summary.md
|-- plots/
|   |-- score_distribution.png
|   `-- selected_counts.png
|-- report/filtering_report.md
`-- topk/10/
    |-- scores.csv
    |-- selected_indices.npy
    |-- metadata.json
    |-- config.yaml
    |-- score_histogram.png
    |-- selected_samples_grid.png
    |-- rejected_samples_grid.png
    |-- best_samples_grid.png
    |-- worst_samples_grid.png
    |-- best_noisy_grid.png
    |-- worst_noisy_grid.png
    |-- best_clean_noisy_grid.png
    `-- worst_clean_noisy_grid.png
```

Success criteria:

- `topk/10/scores.csv` exists;
- `topk/10/selected_indices.npy` exists;
- `summary.json` exists at the experiment root;
- `metrics/filtering_summary.csv` contains the top-k 10 run;
- selected index count is greater than 0;
- log contains no traceback.

## 9. TDnCNN Smoke

Entrypoint audited:

```text
scripts/train_tdncnn.py
scripts/internal/run_TDnCNN_image_suite.py
scripts/internal/tdncnn_image_runs_config.py
scripts/internal/train_TDnCNN_image.py
```

ImageNet-100 configured runs include:

```text
full
topk_10
topk_10_smoke
```

Use the dedicated smoke run:

```bash
python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10_smoke --epochs 1 --batch-size 2 --max-train-samples 16 --max-test-samples 8
```

Expected artifacts:

```text
checkpoints/imagenet100/tdncnn/
`-- topk_10_smoke.pth

experiments/imagenet100/exp_006_tdncnn/topk_10_smoke/
|-- config.json
|-- metrics.json
|-- topk_10_smoke_example.png
`-- results/
    |-- metrics_history.csv
    |-- loss_curve.png
    |-- metrics_grid.png
    |-- per_image_metrics.csv
    |-- qualitative_triplets.png
    `-- error_maps.png
```

Success criteria:

- `checkpoints/imagenet100/tdncnn/topk_10_smoke.pth` exists;
- `metrics.json` exists;
- `results/metrics_history.csv` contains one epoch;
- `results/per_image_metrics.csv` exists;
- log contains no traceback.

## 10. Research Plots Smoke

Entrypoint audited:

```text
scripts/generate_research_plots.py
src/evaluation/research_filtering_plots.py
src/evaluation/research_training_dynamics_plots.py
src/evaluation/research_tdncnn_plots.py
src/evaluation/research_qualitative_plots.py
```

Command:

```bash
python scripts/generate_research_plots.py --dataset imagenet100 --stage all --strict false
```

Expected filtering plot artifacts from the smoke top-k 10 run:

```text
experiments/imagenet100/exp_005_filtering/plots/research_style/
|-- filtering_research_plots_manifest.json
|-- score_distribution_topk_10.png
|-- score_distribution_topk_10.pdf
|-- score_distribution_random_10.png
|-- score_distribution_random_10.pdf
|-- score_distribution_overlay.png
|-- score_distribution_overlay.pdf
|-- selected_vs_rejected_examples.png
`-- selected_vs_rejected_examples.pdf
```

Expected DDPM dynamics artifacts:

```text
experiments/imagenet100/exp_003_latent_ddpm_validation/
|-- metrics/score_training_dynamics.csv
`-- plots/research_style/
    |-- score_training_dynamics.png
    `-- score_training_dynamics.pdf
```

Expected TDnCNN plot artifacts, if `per_image_metrics.csv` exists for
`topk_10_smoke`:

```text
experiments/imagenet100/exp_006_tdncnn/comparison_plots/research_style/
|-- psnr_cdf_comparison.png
|-- psnr_cdf_comparison.pdf
|-- mse_cdf_comparison.png
|-- mse_cdf_comparison.pdf
|-- ssim_cdf_comparison.png
|-- ssim_cdf_comparison.pdf
|-- lpips_cdf_comparison.png
|-- lpips_cdf_comparison.pdf
|-- quality_gap_summary.csv
`-- quality_gap_summary.md
```

Optional qualitative TDnCNN artifacts may be absent in a smoke-only run because
the current qualitative plotting code looks for `full`, `topk_10`, `random_10`,
and `quantile_10` run names, not `topk_10_smoke`.

Success criteria:

- the command reports generated files;
- at least filtering, DDPM dynamics, and TDnCNN comparison plots are generated
  from available smoke artifacts;
- `plots/research_style/` exists where applicable;
- log contains no traceback.

## 11. Smoke Completion Checklist

| Stage | Command | Output Exists | Passed |
| ----- | ------- | ------------- | ------ |
| Pre-flight | `python -m compileall -q scripts src` | n/a | [ ] |
| Smoke data layout | `Test-Path data/smoke_imagenet/train -PathType Container` | class dirs visible | [ ] |
| Encoder smoke | `python scripts/train_encoder.py noise-consistency ... --epochs 1 --fast_dev_run` | `checkpoints/imagenet100/autoencoders/noise_consistency_large_latent256/autoencoder_checkpoint.pt` | [ ] |
| Encoder validation | `python scripts/evaluate_encoder.py compare-encoders ...` | `experiments/imagenet100/exp_002_encoder_validation/summary.json` | [ ] |
| DDPM baseline | `python scripts/main.py ... latent_noise_mode=baseline ... id_label=_smoke` | `checkpoints/imagenet100/ddpm/latent_ddpm_baseline_ae_noise_consistency_imagenet100_smoke/epoch_001.pth` | [ ] |
| DDPM induced | `python scripts/main.py ... latent_noise_mode=induced ... id_label=_smoke` | `checkpoints/imagenet100/ddpm/latent_ddpm_induced_ae_noise_consistency_imagenet100_smoke/epoch_001.pth` | [ ] |
| Score validation | `python scripts/evaluate_latent_ddpm_score.py --dataset imagenet100 ...` | `experiments/imagenet100/exp_003_latent_ddpm_validation/summary.json` | [ ] |
| Filtering | `python scripts/main.py dataset=imagenet100 task=filter_dataset ...` | `experiments/imagenet100/exp_005_filtering/topk/10/selected_indices.npy` | [ ] |
| TDnCNN | `python scripts/train_tdncnn.py --dataset imagenet100 --run topk_10_smoke ...` | `checkpoints/imagenet100/tdncnn/topk_10_smoke.pth` | [ ] |
| Research plots | `python scripts/generate_research_plots.py --dataset imagenet100 --stage all --strict false` | `plots/research_style/` artifacts | [ ] |

## 12. Docker Build Readiness

The ImageNet-100 pipeline is ready for Docker image assembly only if all of the
following are true:

- pre-flight compile check passed;
- `data/smoke_imagenet` exists in ImageFolder layout with `train/` and `val/`;
- `IMAGENET100_ROOT` was set to `data/smoke_imagenet` in the same PowerShell
  session used for the smoke commands;
- smoke class directories are present in both splits;
- smoke image counts are `train=1000` and `val=200`;
- encoder smoke passed;
- encoder validation smoke passed;
- DDPM baseline smoke passed;
- DDPM induced smoke passed;
- score validation smoke passed;
- filtering smoke passed;
- TDnCNN smoke passed;
- research plots smoke passed;
- all expected checkpoints and output directories exist;
- logs from every stage contain no traceback;
- no smoke command had to use unsupported parameters or local code changes.

If any stage fails, do not build the Docker image yet. Fix the failing stage
first and rerun the smoke pipeline from that stage onward.

Passing this runbook does not prove full ImageNet-100 scientific readiness. It
only proves that the existing ImageNet-100 pipeline can execute end-to-end on a
small ImageFolder dataset before containerization.
