import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant",
        nargs="?",
        default="baseline",
        choices=["baseline", "noise-consistency", "representation", "vae"],
    )
    args = parser.parse_args()

    if args.variant == "baseline":
        from scripts.internal import train_autoencoder_baseline_mnist

        train_autoencoder_baseline_mnist.main()
    elif args.variant == "noise-consistency":
        from scripts.internal import train_autoencoder_noise_consistency_mnist

        train_autoencoder_noise_consistency_mnist.main()
    elif args.variant == "representation":
        from scripts.internal import train_autoencoder_representation_mnist

        train_autoencoder_representation_mnist.main()
    elif args.variant == "vae":
        from scripts.internal import train_autoencoder_vae_mnist

        train_autoencoder_vae_mnist.main()


if __name__ == "__main__":
    main()
