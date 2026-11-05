# SIDD evaluation

This experiment evaluates denoising models trained on ImageNet-100 using real noisy images from the SIDD Validation sRGB dataset.

The models are evaluated without additional training on SIDD.

## Dataset

The evaluation uses two files:

- `ValidationNoisyBlocksSrgb.mat`
- `ValidationGtBlocksSrgb.mat`

Place them in `data/sidd/validation/`.

The dataset contains 40 scenes with 32 image blocks per scene, giving 1,280 noisy/clean pairs. Each block has a resolution of 256 × 256 pixels.

The images are loaded as RGB tensors with values in [0, 1]. The evaluation uses the original image blocks without resizing or adding synthetic noise.

The loader supports standard MATLAB MAT files and MATLAB v7.3 files using SciPy and h5py.

## Models and noise settings

The evaluated models are TDnCNN and DRUNet, using checkpoints saved during the ImageNet-100 experiments.

The standalone evaluation compares Full and Top-K subsets at 10%, 20% and 30%. The final external comparison uses DRUNet with all ten training variants and three corpus seeds.

For DRUNet, the primary noise-level input is fixed at:

```text
sigma = 25 / 255
```

This value is passed to DRUNet as its noise-level condition. The original SIDD noisy images are used directly.

The standalone experiment also supports a separate comparison with conditioning levels 5, 10, 15 and 25 (in /255 units).

## Evaluation metrics

For each image block, I calculate:

- PSNR
- SSIM
- MSE
- LPIPS

Predictions are clipped to [0, 1] before evaluation. PSNR is calculated from MSE, and SSIM is computed on RGB images using `data_range=1`.

The results are saved for every image block. For scene-level comparisons, the metrics are first averaged over the 32 blocks of each scene.

The filtered models are compared with Full on the same image blocks. Bootstrap confidence intervals are calculated by sampling scenes.

## Running the evaluation

The configuration file is:

`configs/evaluation/imagenet100_sidd.yaml`

First, check the dataset and trained checkpoints:

```bash
python scripts/run_sidd_evaluation.py --stage preflight --config configs/evaluation/imagenet100_sidd.yaml --sidd-root data/sidd/validation --imagenet-experiment-root <imagenet-run> --output-root experiments/sidd/imagenet100/check
```

To test the evaluation on one image block:

```bash
python scripts/run_sidd_evaluation.py --stage smoke --config configs/evaluation/imagenet100_sidd.yaml --sidd-root data/sidd/validation --imagenet-experiment-root <imagenet-run> --output-root experiments/sidd/imagenet100/one_block --device cuda --max-blocks 1
```

To run the full evaluation:

```bash
python scripts/run_sidd_evaluation.py --stage all --config configs/evaluation/imagenet100_sidd.yaml --sidd-root data/sidd/validation --imagenet-experiment-root <imagenet-run> --output-root experiments/sidd/imagenet100/full --device cuda
```

The results include per-block metrics, scene-level summaries, comparisons between models and figures. Use `--resume` to continue an interrupted evaluation with the same configuration and output directory.

The final three-seed results are aggregated using the scripts described in the main README.