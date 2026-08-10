# Repository Structure

```text
configs/experiments/
  mnist.yaml
  cifar10.yaml
  imagenet100.yaml

scripts/
  run_mnist_pipeline.py
  run_cifar10_pipeline.py
  run_imagenet100_pipeline.py
  run_imagenet100_pipeline.sh
  run_external_evaluation.py

src/
  mnist_pipeline.py
  cifar10_pipeline.py
  imagenet100_pipeline.py
  ddpm/image_space.py
  evaluation/
  utils/

legacy/
  encoder_pipeline/
  latent_ddpm/
  old_analysis/
  old_launchers/
```
