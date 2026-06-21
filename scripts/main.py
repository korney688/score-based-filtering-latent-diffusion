import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from filter_dataset import filter_dataset
from train_DDPM import run_train_DDPM

sys.path.append(os.getcwd())

log = logging.getLogger(__name__)


def experiment_pipe(cfg: DictConfig):
    if cfg.task == "train_latent_DDPM":
        log.info("##### Starting train_latent_DDPM #####")
        log.info(OmegaConf.to_yaml(cfg.train_latent_DDPM))
        run_train_DDPM(cfg.train_latent_DDPM)

    elif cfg.task == "filter_dataset":
        log.info("##### Starting filter_dataset #####")
        log.info(OmegaConf.to_yaml(cfg.filter_dataset))
        filter_dataset(cfg.filter_dataset)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    try:
        experiment_pipe(cfg)
    except Exception as e:
        log.error(f"Critical error in execution: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
