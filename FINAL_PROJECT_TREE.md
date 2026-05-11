Final Project Tree

Generated after controlled cleanup and archival.

Long-running outputs, model checkpoints, experiment artifacts, `.venv`, `.git`, `.idea`, and `__pycache__` are omitted from this tree.

```text
archive_legacy/
  scripts/
    evaluation_pipe.py
    train_TDnCNN_example.py

archive_unused/
  notebooks/
    Gap_dynamics/
      tools_v3.py
  scripts/
    analyze_filtering.py
    analyze_latent_noise.py
    analyze_latent_noise_mismatch.py
    ddpm_baseline_mnist.py
    ddpm_baseline_score_analysis.py
    latent_score_calibration.py
    quick_encoder_score_comparison.py
    sanity_check_score.py
    score_analysis_mnist.py
  src/
    inference_ddpm_image.py
    make_mnist_noisy_h5.py
    train_autoencoder.py
  cleanup_report.md

configs/
  config.yaml
  filter_dataset/
    default.yaml
  hydra/
    hydra_config.yaml
  train_latent_DDPM/
    default.yaml

data/
  MNIST/
    raw/
    clean/
    noisy_gaussian_v1/
  README.md

experiments/
  exp_001_baseline_ddpm/
  exp_002_encoder_validation/
  exp_003_aligned_latent_ddpm/
  exp_004_score_calibration/
  exp_005_filtering/
  exp_006_tdncnn/
  README.md

models/
  autoencoders/
  ddpm/
  tdncnn/
  README.md

notebooks/
  Evaluation_pipe/
    main.ipynb
  Gap_dynamics/
    main.ipynb
    new/
      TDnCNN_model_filtered_QQ_spread_10.csv
      TDnCNN_model_filtered_QQ_spread_50.csv
      TDnCNN_model_filtered_random_10.csv
      TDnCNN_model_filtered_top_k_10_16.csv
      TDnCNN_model_filtered_top_k_50.csv
      TDnCNN_model_v2.csv

scripts/
  evaluate_encoder.py
  evaluate_pipeline.py
  evaluate_score.py
  filter_dataset.py
  internal/
    __init__.py
    run_TDnCNN_image_suite.py
    tdncnn_image_runs_config.py
    train_autoencoder_baseline_mnist.py
    train_autoencoder_noise_consistency_mnist.py
    train_autoencoder_representation_mnist.py
    train_autoencoder_vae_mnist.py
    train_TDnCNN_image.py
  main.py
  train_ddpm.py
  train_encoder.py
  train_tdncnn.py

src/
  autoencoder.py
  autoencoder_noise_consistency.py
  autoencoder_representation.py
  autoencoder_vae.py
  datasets.py
  DDPM_model.py
  filter_mnist_qq.py
  filter_mnist_top_k.py
  filters.py
  tdncnn_datasets.py
  TDnCNN_image.py
  tools.py
  Unet_model.py
  evaluation/
    __init__.py
    encoder_score_validation.py
    encoder_validation.py
    filtering_evaluation.py
    score_calibration.py
    score_validation.py

outputs/
  README.md
  final_results/
    filtering/
      README.md
      qq/
      topk/
  debug/
    filtering/
      README.md
  temporary/
    filtering/
      README.md

ACTIVE_ENTRYPOINTS.md
ACTIVE_FILES.md
FINAL_PROJECT_TREE.md
FINAL_PIPELINE.md
protocol_exp.docx
README.md
requirements.txt
Dockerfile
```

## Notes

- `archive_legacy/` preserves old wireless / Quadriga / complex-signal / TDnCNN 3D entry scripts that still existed in the working tree.
- `archive_unused/` preserves unused exploratory and obsolete scripts instead of deleting them.
- `src/` contains code and reusable modules only; generated filtering artifacts live under `outputs/`.
