import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset_registry import DATASET_SPECS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder_variant",
        nargs="?",
        default="baseline",
        choices=["baseline", "noise-consistency", "representation", "vae"],
    )
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_SPECS))
    parser.add_argument("--variant", dest="noise_consistency_variant", choices=["small", "large"])
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--download", action="store_true")
    args, unknown_args = parser.parse_known_args()
    args.overrides.extend(unknown_args)

    requested_dataset = args.dataset
    for override in args.overrides:
        key, separator, value = override.partition("=")
        if separator == "=" and key == "dataset":
            requested_dataset = value

    if args.encoder_variant != "noise-consistency" and requested_dataset not in {None, "mnist"}:
        parser.error("Only dataset=mnist is supported for non-noise-consistency encoder variants")

    if args.encoder_variant != "noise-consistency" and (
        args.noise_consistency_variant is not None or args.latent_dim is not None
    ):
        parser.error("--variant and --latent-dim are supported only for noise-consistency")

    if args.encoder_variant == "baseline":
        from scripts.internal import train_autoencoder_baseline_mnist

        train_autoencoder_baseline_mnist.main()
    elif args.encoder_variant == "noise-consistency":
        from scripts.internal import train_autoencoder_noise_consistency_mnist

        forwarded_args = list(args.overrides)
        if args.dataset is not None:
            forwarded_args.append(f"dataset={args.dataset}")
        if args.noise_consistency_variant is not None:
            forwarded_args.append(f"variant={args.noise_consistency_variant}")
        if args.latent_dim is not None:
            forwarded_args.append(f"latent_dim={args.latent_dim}")
        if args.epochs is not None:
            forwarded_args.append(f"epochs={args.epochs}")
        if args.data_root is not None:
            forwarded_args.append(f"data_root={args.data_root}")
        if args.fast_dev_run:
            forwarded_args.append("--fast_dev_run")
        if args.download:
            forwarded_args.append("--download")
        train_autoencoder_noise_consistency_mnist.main(forwarded_args)
    elif args.encoder_variant == "representation":
        from scripts.internal import train_autoencoder_representation_mnist

        train_autoencoder_representation_mnist.main()
    elif args.encoder_variant == "vae":
        from scripts.internal import train_autoencoder_vae_mnist

        train_autoencoder_vae_mnist.main()


if __name__ == "__main__":
    main()
