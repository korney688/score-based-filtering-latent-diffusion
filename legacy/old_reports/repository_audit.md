# Repository Audit: Score-Based Filtering in Diffusion Pipelines for Image Denoising

Date: 2026-08-10

Scope: read-only technical audit before cleanup/reorganization. This report is the only file intentionally created by this audit. No files were moved, deleted, renamed, or edited.

## 1. Executive Summary

The repository currently mixes three generations of work:

1. Current image-space variable-AWGN research pipeline for CIFAR-10 and ImageNet-100.
2. Older image-space replication / TDnCNN / DRUNet experiment infrastructure that is still partly useful as reference or external evaluation support.
3. Legacy encoder / autoencoder / latent-DDPM pipeline and older experiment outputs.

Current entry points found:

| Domain | Current entry point | Status | Evidence |
|---|---|---|---|
| CIFAR-10 score validation | `scripts/run_cifar10_score_degradation_validation.py` -> `src/evaluation/cifar10_score_degradation_validation.py` | CURRENT | Imports current CIFAR score-validation module; config/artifacts under `experiments/cifar10_score_degradation_validation_v1`. |
| CIFAR-10 selected filtering | `scripts/run_cifar_selected_filtering.py` | CURRENT | Reads selected epoch/timestep/MC16 artifacts and writes `06_filtering_e18_t100_mc16`. |
| CIFAR-10 TDnCNN downstream | `scripts/run_cifar_selected_downstream.py` | CURRENT | Reads current filtering artifacts and trains/evaluates TDnCNN branches. |
| CIFAR-10 DRUNet downstream | `scripts/run_cifar_selected_drunet.py` | CURRENT, new branch | Reads same current subsets; DRUNet-specific downstream/external branch. |
| CIFAR-10 combined/full50k launcher | `scripts/run_cifar_full50k_pipeline.py` | CURRENT / SPLIT_CANDIDATE | Contains current CIFAR image-space stages plus optional DRUNet dispatch; also has transitional/full50k logic. |
| ImageNet-100 image-space | `scripts/imagenet100_image_space_variable_awgn.py` -> `src/imagenet100_image_space_variable_awgn.py` | CURRENT | Full staged image-space pipeline with prepare/ddpm/score/filter/downstream/eval. |
| ImageNet-100 production shell | `scripts/run_imagenet100_image_space_variable_awgn_full.sh` | CURRENT | Stage graph for image-space variable-AWGN full production run. |
| ImageNet-100 production shell alias/older | `scripts/run_imagenet100_image_space_variable_awgn.sh` | CURRENT / REVIEW_MANUALLY | Similar image-space launcher; verify which is canonical after remote run completes. |
| External validation | `scripts/evaluate_external.py` | CURRENT / SHARED | Supports TDnCNN and DRUNet; used by CIFAR DRUNet runner and ImageNet image-space runner. |

Major reorganization risk:

- `src/DDPM_model.py` is reached by current CIFAR image-space scripts, but imports `src.autoencoder` and `src.autoencoder_noise_consistency` at module import time. This creates a current-to-legacy dependency even when current runtime uses image-space DDPM behavior. Treat as `SPLIT_CANDIDATE`, not simple legacy.
- `src/dataset_registry.py`, `src/tdncnn_datasets.py`, `src/TDnCNN_image.py`, `src/DRUNet_image.py`, `src/Unet_model.py`, and external benchmark helpers are shared by current and older launchers.
- Many current source files are untracked in Git. Cleanup should not start until these untracked files are either committed, intentionally ignored, or explicitly archived.

Safety assessment:

- It is not safe to move whole top-level folders yet.
- It is safe to plan a phased migration after freezing current entry points and splitting `DDPM_model.py` / dataset utilities.

## 2. Current Pipeline

### 2.1 CIFAR-10 Current Image-Space Pipeline

Actual scientific flow:

```text
CIFAR-10 clean
  -> deterministic variable per-image AWGN corpus
  -> image-space DDPM
  -> score degradation validation
  -> selected epoch=18, report timestep=100, code index=99, MC K=16
  -> score computation / filtering
  -> Full / Top-K 10% / Quantile-spread 10% / Random where available
  -> TDnCNN downstream
  -> DRUNet downstream
  -> internal evaluation
  -> external benchmarks
```

Core files:

| Stage | Files | Artifacts |
|---|---|---|
| Corpus / deterministic variable AWGN | `src/image_lab_replication_datasets.py`, `src/evaluation/cifar10_score_degradation_validation.py`, `scripts/run_cifar10_score_degradation_validation.py` | `experiments/cifar10_score_degradation_validation_v1/01_corpus/train_corpus.pt`, `train_manifest.csv` |
| DDPM training / scoring | `src/evaluation/cifar10_score_degradation_validation.py`, `scripts/run_cifar10_score_degradation_validation.py`, `src/DDPM_model.py` | `03_ddpm/checkpoints/epoch_018.pt`; selected config in `05_selected_configuration/selected_score_config.yaml` |
| Filtering | `scripts/run_cifar_selected_filtering.py`, `src/evaluation/image_lab_replication_analysis.py` | `06_filtering_e18_t100_mc16/01_scores/scores_full.csv`, `02_subsets/*.npy` |
| TDnCNN downstream | `scripts/run_cifar_selected_downstream.py`, `src/TDnCNN_image.py`, `src/image_lab_replication_datasets.py` | `07_downstream_e18_t100_mc16/02_training/*/best.pt`, `03_evaluation`, `04_summary` |
| DRUNet downstream | `scripts/run_cifar_selected_drunet.py`, `src/DRUNet_image.py`, `src/external/drunet/*`, `scripts/evaluate_external.py` | `drunet/01_full`, `02_topk_10`, `03_quantile_10`, `04_internal_evaluation`, `05_external_evaluation` |
| External validation | `scripts/evaluate_external.py`, `scripts/run_cifar_selected_drunet.py` | CIFAR DRUNet external outputs under `experiments/cifar10_score_degradation_validation_v1/drunet/05_external_evaluation` after run |

Current CIFAR selected config:

```text
epoch = 18
report timestep = 100
code timestep index = 99
MC K = 16
```

Evidence:

- `configs/experiments/cifar10_full50k_e18_t100_mc16.yaml`
- `configs/experiments/cifar10_drunet_e18_t100_mc16.yaml`
- `experiments/cifar10_score_degradation_validation_v1/06_filtering_e18_t100_mc16/00_protocol/run_meta.json`
- `experiments/cifar10_score_degradation_validation_v1/07_downstream_e18_t100_mc16/00_protocol/resolved_downstream_config.yaml`

### 2.2 ImageNet-100 Current Image-Space Pipeline

Actual flow implemented in `src/imagenet100_image_space_variable_awgn.py`:

```text
prepare
-> ddpm_train
-> score_validate
-> score_select
-> filter
-> tdncnn_full/topk/quantile/random
-> tdncnn_evaluate
-> drunet_full/topk/quantile/random
-> drunet_evaluate
-> summary
```

External benchmarks have been intentionally removed from the remote ImageNet production flow in the current discussion because external benchmarks are evaluated locally after downloading final weights.

Core files:

| Stage | Files | Artifacts |
|---|---|---|
| Python CLI wrapper | `scripts/imagenet100_image_space_variable_awgn.py` | Calls `src.imagenet100_image_space_variable_awgn.main()` |
| Pipeline implementation | `src/imagenet100_image_space_variable_awgn.py` | All staged implementation |
| Production shell | `scripts/run_imagenet100_image_space_variable_awgn_full.sh` | Remote `/workspace` production launcher |
| Config | `configs/experiments/imagenet100_image_space_variable_awgn_v1.yaml` | Production/stage parameters |
| DDPM model | `src/Unet_model.py` via `ImageSpaceDDPM` wrapper | Direct image-space epsilon-prediction DDPM |
| Downstream | `src/TDnCNN_image.py`, `src/DRUNet_image.py` | Model-specific roots under `06_downstream/<model>/<variant>` |
| Shared filtering algorithm | `src/evaluation/image_lab_replication_analysis.py` | Quantile-spread reused from CIFAR implementation |

Important ImageNet config values:

- image size: `64`
- expected train samples: `130000` in config, actual remote preflight reported `120581`
- expected val samples: `5000` in config, actual remote preflight reported `13365`
- DDPM epochs: `40`
- timestep grid: `[1, 2, 5, 10, 20, 50, 100, 200, 500]`
- coarse MC K: `4`
- final MC K: `16`
- TDnCNN max steps: `50000`
- DRUNet max steps: `30480`

Local `experiments/imagenet100_image_space_variable_awgn_v1` contains only smoke/debug runs; the current production run discussed in chat is remote under `/workspace/experiments/.../imagenet100_image_space_variable_awgn_001`.

## 3. Entry Points

| path | type | category | used_by_current_pipeline | evidence | proposed_action | reason |
|---|---|---|---|---|---|---|
| `scripts/run_cifar10_score_degradation_validation.py` | Python CLI | CURRENT | YES | Imports `src.evaluation.cifar10_score_degradation_validation` | KEEP | Current CIFAR score validation entrypoint |
| `src/evaluation/cifar10_score_degradation_validation.py` | Python module | CURRENT | YES | Imported by CIFAR score runner and tests | KEEP | Current CIFAR split/timestep/score-validation implementation |
| `scripts/run_cifar_selected_filtering.py` | Python CLI | CURRENT | YES | Imports DDPM/model, dataset utilities, quantile exact selector | KEEP | Current selected epoch/timestep filtering runner |
| `scripts/run_cifar_selected_downstream.py` | Python CLI | CURRENT | YES | Imports TDnCNN and materialized corpus utilities | KEEP | Current TDnCNN branch |
| `scripts/run_cifar_selected_drunet.py` | Python CLI | CURRENT | YES | Imports DRUNet, TDnCNN checkpoints for combined tables, external helpers | KEEP | Current DRUNet branch |
| `scripts/run_cifar_full50k_pipeline.py` | Python CLI | CURRENT / INTERMEDIATE | YES | Imports selected filtering and has `drunet_*` dispatch | REVIEW_MANUALLY | Contains current and transitional/full50k logic |
| `scripts/run_image_lab_replication.py` | Python CLI | INTERMEDIATE / CURRENT_REFERENCE | INDIRECT | Imported by tests, contains image-space corpus implementation | REVIEW_MANUALLY | Reference implementation for current CIFAR algorithms |
| `scripts/imagenet100_image_space_variable_awgn.py` | Python CLI | CURRENT | YES | Imports `src.imagenet100_image_space_variable_awgn` | KEEP | ImageNet image-space entrypoint |
| `src/imagenet100_image_space_variable_awgn.py` | Python module | CURRENT | YES | Implements full ImageNet staged graph | KEEP | Current ImageNet implementation |
| `scripts/run_imagenet100_image_space_variable_awgn_full.sh` | Bash launcher | CURRENT | YES | Remote production launcher used in active run | KEEP | Current production shell |
| `scripts/run_imagenet100_image_space_variable_awgn.sh` | Bash launcher | CURRENT / DUPLICATE_CANDIDATE | YES/UNKNOWN | Same family as full launcher | REVIEW_MANUALLY | Decide canonical shell after production stabilizes |
| `scripts/evaluate_external.py` | Python CLI/helper | CURRENT / SHARED | YES | Imported by current ImageNet/DRUNet runner; supports TDnCNN/DRUNet | KEEP | External benchmark evaluator |
| `scripts/run_imagenet100_full.sh` | Bash launcher | LEGACY / INTERMEDIATE | NO for new image-space graph | Evidence: old ImageNet full launcher; current graph uses `*_image_space_variable_awgn*` | MOVE_TO_LEGACY after review | Older ImageNet production convention, likely encoder/latent/old downstream |
| `scripts/run_imagenet100_full_noisy.sh` | Bash launcher | LEGACY / INTERMEDIATE | NO for final current graph | No longer current; old noisy pipeline conventions | REVIEW_MANUALLY | May contain useful remote conventions |
| `scripts/run_imagenet100_dgx_smoke.sh` | Bash launcher | SMOKE / LEGACY | NO | DGX smoke launcher for older ImageNet stages | REVIEW_MANUALLY | Keep until smoke strategy consolidated |
| `scripts/run_imagenet100_downstream_grid.sh` | Bash launcher | INTERMEDIATE | NO/UNKNOWN | Uses old TDnCNN grid roots/checkpoints | REVIEW_MANUALLY | May contain result-generation logic |
| `scripts/run_drunet_imagenet100_production.sh` | Bash launcher | LEGACY / INTERMEDIATE | NO for new image-space graph | Uses `scripts/train_drunet.py` old suite | REVIEW_MANUALLY | Old DRUNet ImageNet sigma25 suite |
| `scripts/train_encoder.py` | Python CLI | LEGACY | NO | Imports autoencoder trainers/configs | MOVE_TO_LEGACY | Encoder branch |
| `scripts/evaluate_encoder.py` | Python CLI | LEGACY | NO | Imports `src.evaluation.encoder_validation` | MOVE_TO_LEGACY | Encoder validation branch |
| `scripts/evaluate_latent_ddpm_score.py` | Python CLI | LEGACY | NO | Imports latent score validation | MOVE_TO_LEGACY | Latent score validation branch |
| `scripts/filter_dataset.py` | Python script | LEGACY / SPLIT_CANDIDATE | UNKNOWN | Imports `src.DDPM_model`, `src.filters`; old Stage 3 filter config | REVIEW_MANUALLY | May contain old filtering utilities but not current selected-image pipeline |
| `scripts/train_ddpm.py` | Python script | LEGACY / SPLIT_CANDIDATE | UNKNOWN | Imports `autoencoder_noise_consistency`; latent DDPM config | REVIEW_MANUALLY | Legacy DDPM training but shares DDPM model code |
| `scripts/train_tdncnn.py` | Python CLI | INTERMEDIATE | NO for current CIFAR selected branch | Delegates to internal image suite | REVIEW_MANUALLY | Older TDnCNN suite still useful for ImageNet/legacy |
| `scripts/train_drunet.py` | Python CLI | INTERMEDIATE | NO for current CIFAR selected branch | Delegates to internal DRUNet suite | REVIEW_MANUALLY | Older DRUNet suite; architecture reused |
| `scripts/internal/train_autoencoder_*.py` | Python scripts | LEGACY | NO | Autoencoder / representation / VAE training | MOVE_TO_LEGACY | Encoder/autoencoder branch |
| `scripts/internal/train_TDnCNN_image.py` | Python module | INTERMEDIATE / SHARED | INDIRECT | Old TDnCNN suite; not current selected CIFAR runner | REVIEW_MANUALLY | May be kept as old ImageNet downstream utility |
| `scripts/internal/train_DRUNet_image.py` | Python module | INTERMEDIATE / SHARED | INDIRECT | Old DRUNet clean-target suite; current DRUNet reuses model not runner | REVIEW_MANUALLY | Not current CIFAR protocol |
| `scripts/generate_research_plots.py` | Python CLI | INTERMEDIATE | UNKNOWN | Imports research plotting modules | REVIEW_MANUALLY | Generated paper figures may still be useful |
| `scripts/analyze_image_lab_replication.py` | Python CLI | INTERMEDIATE | UNKNOWN | Analysis of image-lab replication root | REVIEW_MANUALLY | Current-reference analysis but not stage runner |
| `scripts/prepare_external_benchmarks.py` | Python CLI | CURRENT_SUPPORT | INDIRECT | Handles Kodak24/CBSD68/Urban100 structure | KEEP | External dataset prep/check helper |
| `scripts/download_imagenet100_hf.py` | Python CLI | SUPPORT | NO in production | Standalone downloader; current preflight prohibits auto-download | REVIEW_MANUALLY | Keep as manual data prep only |
| `scripts/prepare_smoke_imagenet_from_cifar10.py` | Python CLI | SMOKE | NO | Smoke data helper | REVIEW_MANUALLY | Smoke-only |
| `scripts/regenerate_compact_visual_audit.py` | Python CLI | DEBUG / INTERMEDIATE | UNKNOWN | Reads previous visual audit outputs | REVIEW_MANUALLY | Generated report tool |
| `scripts/main.py` | Python script | LEGACY / UNKNOWN | NO | Old Hydra task launcher | REVIEW_MANUALLY | Superseded by explicit runners |

## 4. Dependency Map

AST-based local import graph from current entrypoints:

```text
scripts.run_cifar10_score_degradation_validation
  -> src.evaluation.cifar10_score_degradation_validation
      -> src.DDPM_model
      -> src.TDnCNN_image
      -> src.dataset_registry
      -> src.image_lab_replication_datasets

scripts.run_cifar_selected_filtering
  -> src.DDPM_model
  -> src.dataset_registry
  -> src.evaluation.image_lab_replication_analysis
  -> src.image_lab_replication_datasets

scripts.run_cifar_selected_downstream
  -> src.TDnCNN_image
  -> src.image_lab_replication_datasets

scripts.run_cifar_selected_drunet
  -> scripts.evaluate_external
  -> src.DRUNet_image
  -> src.TDnCNN_image
  -> src.image_lab_replication_datasets

scripts.imagenet100_image_space_variable_awgn
  -> src.imagenet100_image_space_variable_awgn
      -> scripts.evaluate_external
      -> src.DRUNet_image
      -> src.TDnCNN_image
      -> src.Unet_model
      -> src.dataset_registry
      -> src.evaluation.image_lab_replication_analysis
      -> src.image_lab_replication_datasets

scripts.evaluate_external
  -> src.DRUNet_image
  -> src.TDnCNN_image

src.DRUNet_image
  -> src.external.drunet.network_unet
      -> relative DPIR blocks in src.external.drunet.basicblock (not fully captured by simple absolute-import graph)
```

Current-to-legacy dependency observed:

```text
current CIFAR scripts
  -> src.DDPM_model
      -> src.autoencoder
      -> src.autoencoder_noise_consistency
```

Classification: `src/DDPM_model.py` is `SPLIT_CANDIDATE`. The file is used by current CIFAR score/filtering code, but imports legacy autoencoder modules. Do not move all autoencoder code until DDPM image-space use is separated or imports are made lazy.

Modules not reached from current entrypoints and likely legacy/intermediate:

- `src/evaluation/encoder_validation.py`
- `src/evaluation/latent_ddpm_score_validation.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`
- `src/datasets.py`
- `src/filters.py`
- `src/tools.py`
- research plotting modules under `src/evaluation/research_*`
- old internal image-suite runners under `scripts/internal/`

Use `REVIEW_MANUALLY` for any module that may be dynamically loaded by config or shell.

## 5. File Classification

| path | type | category | used_by_current_pipeline | evidence | proposed_action | reason |
|---|---|---|---|---|---|---|
| `README.md` | docs | INTERMEDIATE | INDIRECT | Mentions old DRUNet `/workspace/data` commands and Docker examples | REVIEW_MANUALLY | Needs update after structure decision |
| `docs/PIPELINE.md` | docs | LEGACY / INTERMEDIATE | UNKNOWN | Existing pipeline doc predates new ImageNet image-space work | REVIEW_MANUALLY | Compare against current graph |
| `docs/imagenet100_image_space_variable_awgn_audit.md` | docs | CURRENT | YES | ImageNet image-space audit document | KEEP | Supports current ImageNet branch |
| `docs/drunet_*` | docs | INTERMEDIATE | UNKNOWN | DRUNet import/integration/preflight/runbook docs | REVIEW_MANUALLY | Some useful, some superseded by current CIFAR DRUNet branch |
| `docs/ENCODERS.md` | docs | LEGACY | NO | Encoder documentation | MOVE_TO_LEGACY | Encoder branch |
| `docs/noisy_imagenet100_worklog_2026-07-03.md` | docs | LEGACY / INTERMEDIATE | UNKNOWN | Old noisy ImageNet worklog | REVIEW_MANUALLY | May explain remote scripts |
| `src/imagenet100_image_space_variable_awgn.py` | source | CURRENT | YES | Current ImageNet full staged implementation | KEEP | Needed for current ImageNet |
| `src/image_lab_replication_datasets.py` | source | CURRENT | YES | Deterministic AWGN corpus/task/eval dataset utilities | KEEP | Needed for current CIFAR and ImageNet |
| `src/evaluation/cifar10_score_degradation_validation.py` | source | CURRENT | YES | Current CIFAR score validation | KEEP | Needed for CIFAR |
| `src/evaluation/image_lab_replication_analysis.py` | source | CURRENT | YES | Exact Top-K/Quantile/Random helpers | KEEP | Used by CIFAR/ImageNet filtering |
| `src/TDnCNN_image.py` | source | CURRENT / SHARED | YES | Downstream TDnCNN model and LPIPS helpers | KEEP | Needed by current downstream/eval |
| `src/DRUNet_image.py` | source | CURRENT / SHARED | YES | Official DRUNet adapter with sigma conditioning | KEEP | Needed by current DRUNet downstream/eval |
| `src/external/drunet/network_unet.py` | source | CURRENT / THIRD_PARTY | YES | Official DRUNet network dependency | KEEP | Needed by DRUNet |
| `src/external/drunet/basicblock.py` | source | CURRENT / THIRD_PARTY | INDIRECT | Relative dependency of DPIR network | KEEP | Needed by DRUNet |
| `src/Unet_model.py` | source | CURRENT / SHARED | YES | ImageNet image-space DDPM UNet | KEEP | Needed by ImageNet and some old scripts |
| `src/dataset_registry.py` | source | SHARED | YES | Used by current ImageNet/CIFAR and legacy scripts | SPLIT_CANDIDATE | Shared dataset utility |
| `src/DDPM_model.py` | source | SPLIT_CANDIDATE | YES for CIFAR | Current CIFAR imports it; file imports autoencoders | SPLIT_CANDIDATE | Separate image-space DDPM from latent/autoencoder code later |
| `src/autoencoder.py` | source | LEGACY / INDIRECT | INDIRECT via `DDPM_model` | Only current reachability is through import side effect | REVIEW_MANUALLY | Move only after DDPM split |
| `src/autoencoder_noise_consistency.py` | source | LEGACY / INDIRECT | INDIRECT via `DDPM_model` | Imported by `DDPM_model` and old trainers | REVIEW_MANUALLY | Move only after DDPM split |
| `src/autoencoder_representation.py` | source | LEGACY | NO | Used by encoder validation / autoencoder trainer | MOVE_TO_LEGACY | Representation autoencoder branch |
| `src/autoencoder_vae.py` | source | LEGACY | NO | Used by VAE trainer / encoder validation | MOVE_TO_LEGACY | VAE branch |
| `src/evaluation/encoder_validation.py` | source | LEGACY | NO | Imported by `scripts/evaluate_encoder.py` | MOVE_TO_LEGACY | Encoder validation |
| `src/evaluation/latent_ddpm_score_validation.py` | source | LEGACY | NO | Imported by `scripts/evaluate_latent_ddpm_score.py` | MOVE_TO_LEGACY | Latent DDPM score validation |
| `src/filters.py` | source | LEGACY / UNKNOWN | NO from current selected graph | Used by `scripts/filter_dataset.py` and tests | REVIEW_MANUALLY | Old filtering API; current exact implementation is elsewhere |
| `src/tdncnn_datasets.py` | source | INTERMEDIATE / SHARED | UNKNOWN | Used by old TDnCNN/DRUNet image suites | REVIEW_MANUALLY | May be needed for old ImageNet downstream |
| `src/datasets.py` | source | LEGACY | NO | Imported by old `train_ddpm.py` | REVIEW_MANUALLY | Legacy dataset H5 path |
| `src/tools.py` | source | LEGACY / UNKNOWN | NO from current graph | Used by old DDPM/filter scripts | REVIEW_MANUALLY | Utility may contain reusable helpers |
| `src/evaluation/research_*.py` | source | INTERMEDIATE | UNKNOWN | Imported by `scripts/generate_research_plots.py` | REVIEW_MANUALLY | Figure-generation support |
| `configs/experiments/imagenet100_image_space_variable_awgn_v1.yaml` | config | CURRENT | YES | Current ImageNet config | KEEP | Needed for production/smoke |
| `configs/experiments/cifar10_full50k_e18_t100_mc16.yaml` | config | CURRENT / INTERMEDIATE | YES | Current selected CIFAR TDnCNN/full50k config | KEEP | Needed for current CIFAR follow-up |
| `configs/experiments/cifar10_drunet_e18_t100_mc16.yaml` | config | CURRENT | YES | Current CIFAR DRUNet branch config | KEEP | Needed for DRUNet |
| `configs/experiments/image_lab_replication/*.yaml` | config | CURRENT_REFERENCE / INTERMEDIATE | INDIRECT | Used by reference image-lab replication tests/runner | REVIEW_MANUALLY | Reference implementation, not final current launch |
| `configs/train_latent_DDPM/default.yaml` | config | LEGACY | NO | Contains `latent_noise_mode`, encoder checkpoint path | MOVE_TO_LEGACY | Latent DDPM branch |
| `configs/filter_dataset/default.yaml` | config | LEGACY / SPLIT_CANDIDATE | NO/UNKNOWN | Uses `ddpm_branch`, `autoencoder_checkpoint_path` | REVIEW_MANUALLY | Old Stage 3 filtering config |
| `configs/dataset/cifar10.yaml` | config | SHARED | YES | Used by current/old dataset registry | KEEP | Dataset config |
| `configs/dataset/imagenet100.yaml` | config | SHARED / CURRENT | YES | Used by ImageNet current and old scripts | KEEP | Dataset config; contains current paths/settings |
| `configs/dataset/mnist*.yaml` | config | POC / LEGACY | NO for main research | REVIEW_MANUALLY | MNIST proof-of-concept / noisy variants |
| `archive_legacy/**` | source/archive | GENERATED/LEGACY | NO | Already archived, ignored by `.gitignore` | REVIEW_MANUALLY | Validate before final archive deletion/move |
| `archive_unused/**` | source/archive | LEGACY | NO | Contains latent/noisy old scripts | REVIEW_MANUALLY | Already outside current graph |
| `scripts/__pycache__/**`, `src/**/__pycache__/**`, `tests/__pycache__/**` | generated | GENERATED | NO | Python bytecode cache | DELETE recommended later | Generated files; do not delete in audit |

## 6. Config Audit

| config | category | domain | evidence | proposed_action |
|---|---|---|---|---|
| `configs/experiments/imagenet100_image_space_variable_awgn_v1.yaml` | CURRENT | ImageNet-100 | Contains image-space variable AWGN, score-validation scan, TDnCNN/DRUNet settings | KEEP |
| `configs/experiments/cifar10_full50k_e18_t100_mc16.yaml` | CURRENT / INTERMEDIATE | CIFAR-10 | Fixed epoch 18/t100/MC16 selected config and TDnCNN comparison | KEEP |
| `configs/experiments/cifar10_drunet_e18_t100_mc16.yaml` | CURRENT | CIFAR-10 | DRUNet branch, same selected subsets | KEEP |
| `configs/experiments/image_lab_replication/cifar10.yaml` | CURRENT_REFERENCE | CIFAR-10 | Reference variable-AWGN replication | REVIEW_MANUALLY |
| `configs/experiments/image_lab_replication/mnist.yaml` | POC | MNIST | Proof-of-concept replication | REVIEW_MANUALLY |
| `configs/train_latent_DDPM/default.yaml` | LEGACY | all/latent | Explicit `latent_noise_mode`, `autoencoder_checkpoint_path` | MOVE_TO_LEGACY |
| `configs/filter_dataset/default.yaml` | LEGACY / SPLIT_CANDIDATE | all/latent filtering | Explicit `ddpm_branch`, `autoencoder_kind`, old quantile fields | REVIEW_MANUALLY |
| `configs/config.yaml` | LEGACY / Hydra | MNIST default latent task | `task: train_latent_DDPM` | MOVE_TO_LEGACY after Hydra status review |
| `configs/dataset/*.yaml` | SHARED | datasets | Used by dataset registry and old scripts | KEEP / REVIEW per dataset |

Absolute local paths in configs were not found in the targeted config scan. `/workspace` paths appear in shell launchers, Dockerfile, and README examples.

## 7. Experiments Audit

Top-level experiment roots:

| root | approx size | files | category | evidence | proposed_action |
|---|---:|---:|---|---|---|
| `experiments/cifar10_score_degradation_validation_v1` | 1767 MB | 395 | CURRENT | Contains current CIFAR corpus, DDPM, selected config, filtering, TDnCNN, DRUNet branch | KEEP / ARCHIVE_FINAL_COPY later |
| `experiments/cifar10_score_degradation_validation_v1/06_filtering_e18_t100_mc16` | 14.7 MB | 202 | CURRENT_FINAL | `scores_full.csv`, subset indices for epoch18/t100/MC16 | KEEP |
| `experiments/cifar10_score_degradation_validation_v1/07_downstream_e18_t100_mc16` | 17.9 MB | 42 | CURRENT_FINAL | TDnCNN Full/Top-K/Quantile outputs | KEEP |
| `experiments/cifar10_score_degradation_validation_v1/drunet` | 623 MB | 8 | CURRENT_IN_PROGRESS | DRUNet branch root; full run started by user | KEEP |
| `experiments/cifar10_score_degradation_validation_v1/drunet_smoke` | 11 MB | 43 | SMOKE | Smoke DRUNet outputs | ARCHIVE or DELETE later |
| `experiments/imagenet100_image_space_variable_awgn_v1` | 82 MB | 565 | SMOKE / LOCAL_DEBUG | Local synthetic/smoke/debug runs only | REVIEW_MANUALLY |
| `experiments/image_lab_replication_v1` | 1200 MB | 240 | CURRENT_REFERENCE / INTERMEDIATE | Earlier image-space variable-AWGN reference, used by CIFAR configs/eval cache | KEEP until current CIFAR fully self-contained |
| `experiments/cifar10` | 20 MB | 158 | LEGACY | Contains `exp_002_encoder_validation`, `exp_003_latent_ddpm_validation`, old filtering/TDnCNN | MOVE_TO_LEGACY/ARCHIVE later |
| `experiments/imagenet100` | 157 MB | 456 | LEGACY / INTERMEDIATE | Encoder validation, latent DDPM, old filtering, old TDnCNN/DRUNet sigma25 suites | ARCHIVE / REVIEW_MANUALLY |
| `experiments/mnist` | 40 MB | 382 | POC / LEGACY | MNIST proof-of-concept and old stages | ARCHIVE as POC |
| `experiments/external_benchmarks` | 140 MB | 226 | INTERMEDIATE / EXTERNAL_RESULTS | Old external outputs for sigma25 variants | REVIEW_MANUALLY |
| `experiments/external_benchmarks_noisy_pg` | 42 MB | 70 | LEGACY / INTERMEDIATE | Noisy PG external outputs | REVIEW_MANUALLY |
| `experiments/_smoke_*` | 0-8 MB | varied | SMOKE | Smoke encoder/filter/score outputs | DELETE recommended later |
| `experiments/noisy_dataset` | 3.7 MB | 10 | INTERMEDIATE | Noisy dataset experiment | REVIEW_MANUALLY |

Duplicate assessment:

- `experiments/cifar10_score_degradation_validation_v1/07_filtering_preview` and `08_downstream_commands` appear intermediate relative to `06_filtering_e18_t100_mc16` and `07_downstream_e18_t100_mc16`. Mark `REVIEW_MANUALLY`, not duplicate, because they may contain generated command/report context.
- Multiple ImageNet smoke roots (`local_smoke_*`, `inline*`) are likely duplicated debug/smoke runs, but no deletion should occur without comparing config hashes and stage markers.

## 8. Artifacts Audit

| artifact group | category | evidence | proposed_action |
|---|---|---|---|
| `experiments/cifar10_score_degradation_validation_v1/01_corpus/train_corpus.pt`, `train_manifest.csv` | CURRENT_FINAL | Current deterministic variable-AWGN corpus | KEEP |
| `03_ddpm/checkpoints/epoch_018.pt` under CIFAR root | CURRENT_FINAL | Selected DDPM checkpoint | KEEP |
| `05_selected_configuration/selected_score_config.yaml` | CURRENT_FINAL | Selected epoch/timestep/MC config | KEEP |
| `06_filtering_e18_t100_mc16/01_scores/scores_full.csv` | CURRENT_FINAL | Full-corpus score table | KEEP |
| `06_filtering_e18_t100_mc16/02_subsets/*.npy` | CURRENT_FINAL | Full/Top-K/Quantile/Random selected indices | KEEP |
| `07_downstream_e18_t100_mc16/02_training/*/best.pt` | CURRENT_FINAL | TDnCNN current checkpoints | KEEP |
| `drunet/01_full`, `02_topk_10`, `03_quantile_10` | CURRENT_IN_PROGRESS | DRUNet current branch | KEEP |
| `experiments/external_benchmarks/*_sigma25` | EXTERNAL_ARTIFACT / INTERMEDIATE | Kodak24/CBSD68/Urban100 metrics/panels | REVIEW_MANUALLY |
| `experiments/cifar10/exp_002_*`, `exp_003_*` | LEGACY_ARTIFACT | Encoder/latent score validation names and folder structure | ARCHIVE |
| `experiments/imagenet100/exp_002_*`, `exp_003_*` | LEGACY_ARTIFACT | Encoder/latent validation | ARCHIVE |
| `experiments/*smoke*` | DEBUG_ARTIFACT | Smoke/debug names | DELETE recommended later after verifying no unique report |
| `docs/*.docx`, `protocol_exp.docx` | INTERMEDIATE_REPORT | Non-code report artifacts, ignored by `.gitignore` | REVIEW_MANUALLY |

## 9. Tests Audit

Tests were analyzed statically only. No test suite was run during this audit to avoid creating new `__pycache__`, temp artifacts, downloads, or training outputs.

| test | category | coverage | risk |
|---|---|---|---|
| `tests/test_filters.py` | LEGACY / CURRENT? | Tests `src.filters` Top-K/Quantile spread old API | Low runtime risk; relevance needs review because current exact selector is in `image_lab_replication_analysis.py` |
| `tests/test_cifar10_score_degradation_validation.py` | CURRENT | CIFAR stratified split and report timestep indexing | Low |
| `tests/test_image_lab_replication.py` | CURRENT_REFERENCE | Deterministic variable-AWGN corpus, exact selectors, eval noise, initial state reuse | Writes temp dirs only if run; useful current reference |
| `tests/test_imagenet100_image_space_variable_awgn.py` | CURRENT | Synthetic ImageNet pipeline components, score select pass/fail, filtering | Writes temp dirs only if run; useful |
| `tests/smoke_imagenet100_image_space_variable_awgn.py` | SMOKE | Smoke command/helper | Review before running |

Existing `tests/__pycache__` is generated and should be removed in cleanup phase, not now.

## 10. Hardcoded Paths

Targeted scan found:

| path | file:line | category | interpretation |
|---|---|---|---|
| `/workspace/data`, `/workspace/experiments` | `scripts/run_imagenet100_image_space_variable_awgn*.sh` | CURRENT_RUNTIME_DEFAULT | Remote Docker defaults; acceptable if overridable |
| `/workspace/data/external_benchmarks` | `scripts/run_imagenet100_image_space_variable_awgn.sh` | CURRENT/OLD_EXTERNAL | Review because ImageNet external benchmarks are now local-only for final weights |
| `/workspace` | `Dockerfile` | DOCKER_RUNTIME | Correct Docker workdir |
| `/workspace/data` | `README.md` old DRUNet commands | DOC_EXAMPLE / LEGACY | Update docs later |
| `/workspace` | `scripts/run_imagenet100_full_noisy.sh`, `run_imagenet100_dgx_smoke.sh` | LEGACY_REMOTE_DEFAULT | Old remote launchers |

No `E:\`, `C:\`, `/home`, or `/mnt` hardcoded paths were found in the targeted source/config scan, except paths inside generated experiment artifacts and chat-derived output files not intended as code.

## 11. Git Hygiene

Current git state at audit start:

- Modified tracked files include README, ImageNet config, external evaluator, old internal TDnCNN/DRUNet scripts, `src/DDPM_model.py`, dataset registry, encoder/latent evaluation modules, and `src/tdncnn_datasets.py`.
- Untracked current pipeline files include CIFAR selected runners, ImageNet image-space runner, new tests, and current image-space modules.
- `git ls-files` count: 67 tracked files.
- `git ls-files --others --exclude-standard` count: 22 untracked files.
- Largest tracked files are small source/docs files; no large tracked checkpoints or experiment outputs were detected by `git ls-files`.

`.gitignore` already excludes:

- `__pycache__/`, `*.pyc`
- `.venv/`, `.env`
- `data/`, `checkpoints/`, `experiments/`, `outputs/`
- `*.npy`, `*.csv`, `*.pt` via broader generated patterns
- media/report artifacts such as `*.png`, `*.docx`, `*.pdf`
- `archive_legacy/`, `archive_unused/`

Potential hygiene issue:

- Because `experiments/` and `docs/*.md` rules interact with ignored/generated policies, final documentation/artifact policy should be clarified before moving reports.
- Existing ignored `__pycache__` files are present in working tree but not tracked.

## 12. Shared-Code Risks

| file | current functions | legacy functions | risk | later refactor |
|---|---|---|---|---|
| `src/DDPM_model.py` | CIFAR image-space DDPM/scoring code reaches this file | Imports autoencoder and noise-consistency code | High | Split `src/ddpm/image_space.py` and `src/ddpm/latent.py`; remove import side effects |
| `src/dataset_registry.py` | ImageNet/CIFAR dataset loading and runtime overrides | Old noisy/encoder dataset modes | Medium | Split dataset specs from legacy corruption modes |
| `src/tdncnn_datasets.py` | Old TDnCNN/DRUNet image suite loaders | Legacy filtered-index and corruption handling | Medium | Keep until old ImageNet suite either archived or ported |
| `scripts/evaluate_external.py` | Current TDnCNN/DRUNet external helpers | Also standalone old DRUNet CLI | Medium | Split helpers from CLI; keep model loading adapters |
| `src/evaluation/image_lab_replication_analysis.py` | Current exact Top-K/Quantile/Random selectors | Reference-analysis naming | Low | Move to `src/filtering/selection.py` later |
| `src/image_lab_replication_datasets.py` | Current deterministic corpus and eval-noise utilities | Name tied to old replication | Low | Move to `src/datasets/variable_awgn.py` later |

## 13. Proposed Final Repository Tree

Minimal-risk target tree:

```text
repository/
├── README.md
├── src/
│   ├── datasets/
│   │   ├── registry.py
│   │   └── variable_awgn.py
│   ├── ddpm/
│   │   ├── image_space.py
│   │   ├── unet.py
│   │   └── latent.py              # only if legacy kept as package
│   ├── filtering/
│   │   └── selection.py
│   ├── downstream/
│   │   ├── tdncnn.py
│   │   └── drunet.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── score_validation.py
│   │   └── external.py
│   └── external/
│       └── drunet/
├── scripts/
│   ├── run_cifar10_pipeline.py
│   ├── run_imagenet100_pipeline.py
│   ├── run_cifar10_drunet.py
│   ├── run_external_validation.py
│   └── internal/
├── configs/
│   ├── experiments/
│   │   ├── cifar10_image_space_variable_awgn.yaml
│   │   └── imagenet100_image_space_variable_awgn.yaml
│   ├── downstream/
│   ├── external_eval/
│   └── smoke/
├── tests/
├── docs/
│   ├── methodology.md
│   ├── experiment_protocol.md
│   ├── cifar10_results.md
│   ├── imagenet100_results.md
│   ├── external_benchmarks.md
│   └── repository_structure.md
├── artifacts_manifest/
│   ├── cifar10_current.yaml
│   └── imagenet100_current.yaml
└── legacy/
    ├── README.md
    ├── encoder_pipeline/
    ├── latent_ddpm/
    ├── old_configs/
    ├── old_reports/
    └── old_experiments_manifest.md
```

Do not move current experiment outputs into Git. Instead create manifests pointing to external/local experiment roots.

## 14. Proposed Migration Plan

Phase 0 - freeze current state:

- Decide whether to commit or otherwise snapshot untracked current files.
- Record current remote ImageNet run command and RUN_ID.
- Save hashes of selected CIFAR artifacts and current configs.

Phase 1 - isolate obvious legacy:

- Move only clearly unreachable encoder/latent scripts/configs/docs to `legacy/`.
- Do not move `DDPM_model.py`, `dataset_registry.py`, or shared downstream modules yet.

Phase 2 - split shared modules:

- Split `src/DDPM_model.py` into image-space and latent components.
- Split deterministic variable-AWGN utilities from `image_lab_replication_*` names.
- Split external evaluator helpers from CLI.

Phase 3 - normalize configs:

- Rename current CIFAR/ImageNet configs to stable names.
- Move smoke configs into `configs/smoke/`.
- Move latent configs into `legacy/old_configs/`.

Phase 4 - normalize launchers:

- Make one canonical CIFAR launcher.
- Make one canonical ImageNet launcher.
- Make one canonical local external benchmark launcher.
- Keep old shell files as legacy runbooks until replaced.

Phase 5 - archive experiments:

- Create manifests for final CIFAR, ImageNet, and external results.
- Archive legacy experiments by root and category.
- Only delete smoke/debug outputs after manual confirmation.

Phase 6 - update docs:

- Replace README current-run instructions.
- Create methodology and current artifact map.

Phase 7 - update `.gitignore`:

- Clarify docs policy.
- Keep experiment/checkpoint/data ignores.

Phase 8 - run tests/smoke:

- Run static import checks.
- Run unit tests that use temp dirs only.
- Run synthetic CIFAR/ImageNet smoke.
- Do not run full training.

## 15. Files Requiring Manual Decision

Highest-risk decisions:

1. `src/DDPM_model.py` - current CIFAR uses it, but it imports autoencoder modules.
2. `src/dataset_registry.py` - shared by old and current paths.
3. `scripts/run_cifar_full50k_pipeline.py` - current plus transitional logic.
4. `scripts/run_imagenet100_image_space_variable_awgn.sh` vs `scripts/run_imagenet100_image_space_variable_awgn_full.sh` - choose canonical launcher.
5. `scripts/evaluate_external.py` - current helper plus standalone CLI; split later.
6. `experiments/image_lab_replication_v1` - still referenced by current CIFAR downstream/eval cache.
7. `experiments/cifar10_score_degradation_validation_v1/drunet` - current DRUNet run in progress; do not touch.
8. `experiments/imagenet100_image_space_variable_awgn_v1` local roots - mostly smoke/debug, but useful as local validation evidence.
9. `scripts/internal/train_TDnCNN_image.py` and `scripts/internal/train_DRUNet_image.py` - older suites, possibly still useful for old ImageNet comparisons.
10. `docs/drunet_*` - mixed preflight/import/runbook docs; consolidate after final DRUNet results.

## 16. Direct Answers

Files needed to reproduce the modern image-space experiment:

- CIFAR core: `scripts/run_cifar10_score_degradation_validation.py`, `src/evaluation/cifar10_score_degradation_validation.py`, `scripts/run_cifar_selected_filtering.py`, `scripts/run_cifar_selected_downstream.py`, `scripts/run_cifar_selected_drunet.py`, `src/image_lab_replication_datasets.py`, `src/evaluation/image_lab_replication_analysis.py`, `src/DDPM_model.py`, `src/TDnCNN_image.py`, `src/DRUNet_image.py`, `src/external/drunet/*`, `scripts/evaluate_external.py`, configs under `configs/experiments/cifar10_*e18_t100_mc16.yaml`.
- ImageNet core: `scripts/imagenet100_image_space_variable_awgn.py`, `src/imagenet100_image_space_variable_awgn.py`, `scripts/run_imagenet100_image_space_variable_awgn_full.sh`, `configs/experiments/imagenet100_image_space_variable_awgn_v1.yaml`, `src/Unet_model.py`, `src/TDnCNN_image.py`, `src/DRUNet_image.py`, `src/evaluation/image_lab_replication_analysis.py`, `scripts/evaluate_external.py` for local external evaluation.

Files that exist because the older project used encoder + latent DDPM:

- `scripts/train_encoder.py`
- `scripts/evaluate_encoder.py`
- `scripts/evaluate_latent_ddpm_score.py`
- `scripts/internal/train_autoencoder_baseline_mnist.py`
- `scripts/internal/train_autoencoder_noise_consistency_mnist.py`
- `scripts/internal/train_autoencoder_representation_mnist.py`
- `scripts/internal/train_autoencoder_vae_mnist.py`
- `src/evaluation/encoder_validation.py`
- `src/evaluation/latent_ddpm_score_validation.py`
- `src/autoencoder_representation.py`
- `src/autoencoder_vae.py`
- `configs/train_latent_DDPM/default.yaml`
- `configs/config.yaml`
- most `experiments/*/exp_002_encoder_validation*` and `experiments/*/exp_003_latent_ddpm_validation*`

Conditional legacy, requiring split/review before moving:

- `src/DDPM_model.py`
- `src/autoencoder.py`
- `src/autoencoder_noise_consistency.py`
- `configs/filter_dataset/default.yaml`
- `scripts/filter_dataset.py`
- `scripts/train_ddpm.py`

NO FILES WERE DELETED OR MOVED.
