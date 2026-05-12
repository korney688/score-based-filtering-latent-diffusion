# Final Project Tree

Generated after validation cleanup. Long-running outputs, model checkpoints, experiment artifacts, `.venv`, `.git`, `.idea`, and `__pycache__` are omitted.

```text
archive_legacy/
archive_unused/

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
  Gap_dynamics/

scripts/
  evaluate_encoder.py
  evaluate_latent_ddpm_score.py
  evaluate_pipeline.py
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
    encoder_validation.py
    filtering_evaluation.py
    latent_ddpm_score_validation.py
    score_calibration.py

outputs/
  README.md

ACTIVE_ENTRYPOINTS.md
ACTIVE_FILES.md
FINAL_PROJECT_TREE.md
FINAL_PIPELINE.md
STAGE2_AB_VALIDATION_AUDIT.md
protocol_exp.docx
README.md
requirements.txt
Dockerfile
```

## Notes

- `scripts/evaluate_latent_ddpm_score.py` is the only active entrypoint for frozen-encoder latent-DDPM score validation.
