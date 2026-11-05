# SIDD Validation sRGB evaluation

Evaluate trained ImageNet-100 denoisers zero-shot on native SIDD noisy/GT pairs. No training, finetuning, noise estimation or SIDD-specific checkpoint selection is performed.

## Data and metrics

Provide `data/sidd/validation/ValidationNoisyBlocksSrgb.mat` and `ValidationGtBlocksSrgb.mat` separately. Automatic download is not implemented. The full workload is 40 scenes x 32 blocks = 1280 paired RGB blocks of 256x256 pixels.

The loader uses scipy.io.loadmat for standard MAT files and h5py for MATLAB v7.3/HDF5. The usual raw shape (40,32,256,256,3) becomes [scene,block,channel,height,width]. Blocks are float32 RGB [0,1]. No resize, crop, added AWGN, gamma transform or ImageNet normalization is applied. Predictions are clamped to [0,1].

Per-block metrics are MSE, PSNR=10*log10(1/MSE), SSIM with data_range=1 and channel_axis=0, and LPIPS using the shared prepare_for_lpips helper. PSNR is infinite for identical images. Full analysis requires finite metrics; when LPIPS is enabled, its model/weights must be available. Disabled LPIPS is recorded as missing.

## Checkpoints and noise conditioning

Read architecture settings from `<imagenet-run>/00_protocol/resolved_config.yaml` and checkpoints from `06_downstream/<model>/<variant>/best.pt`. A missing resolved snapshot can fall back to --imagenet-config-path with a recorded warning. Preflight checks model/variant identity, true-best metadata, source/checkpoint hashes and strict state_dict loading. Checkpoints stay in their source root.

The standalone config evaluates TDnCNN and DRUNet Full and Top-K at 10%, 20%, 30%. Primary DRUNet conditioning is sigma=25/255. Its separate sensitivity analysis uses {5,10,15,25}/255, without tuning sigma against GT. The final publication script instead evaluates DRUNet across all ten variants and three corpus seeds, using only the fixed primary condition.

## Run and resume

Inspect an existing source run before evaluation:

```sh
python scripts/run_sidd_evaluation.py --stage preflight --config configs/evaluation/imagenet100_sidd.yaml --sidd-root data/sidd/validation --imagenet-experiment-root <imagenet-run> --output-root experiments/sidd/imagenet100/check
```

For a real one-block smoke, use a separate output root:

```sh
python scripts/run_sidd_evaluation.py --stage smoke --config configs/evaluation/imagenet100_sidd.yaml --sidd-root data/sidd/validation --imagenet-experiment-root <imagenet-run> --output-root experiments/sidd/imagenet100/one_block --device cuda --max-blocks 1
```

Run the full standalone evaluation with a new output root:

```sh
python scripts/run_sidd_evaluation.py --stage all --config configs/evaluation/imagenet100_sidd.yaml --sidd-root data/sidd/validation --imagenet-experiment-root <imagenet-run> --output-root experiments/sidd/imagenet100/full --device cuda
```

Add --resume only when continuing the same workload/config/source identity. Output roots must be outside the source experiment root. Markers bind the resolved configs, input MAT files, checkpoint, workload mode, max_blocks and expected block count. A one-block smoke marker cannot resume the full evaluation.

## Analysis and outputs

Analysis reads only expected result shards with valid markers. The full standalone grid needs 1280 baseline rows and 1280 rows per model/variant/sigma, unique global indices 0..1279, coverage of all 40 scenes and 32 blocks, and no missing combinations. Scientific analysis fails on incomplete observations or non-finite required metrics.

Outputs include `00_protocol/` configs, input/checkpoint manifests and dataset audit; `02_evaluation/` per-combination block metrics; `03_analysis/` combined block and scene metrics, paired comparisons, sensitivity results and primary tables; `04_figures/`; and `05_summary/` summaries. Runtime metadata excludes credentials. Keep complete bundles private because provenance can contain source paths.

For scene-level comparisons, first average the 32 blocks per scene, then compute each variant minus Full over 40 paired scene observations. Bootstrap intervals resample scenes with a fixed seed. Higher PSNR/SSIM and lower LPIPS/MSE are better. The final three-seed publication tables follow the aggregation protocol in the [experiment protocol](experiment_protocol.md#evaluation-and-aggregation).
