import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from preprocessing_pipe import preprocessing_pipe
from generate_noisy_dataset import generate_noisy_dataset
from train_DDPM import run_train_DDPM, run_finetune_DDPM
from train_TDnCNN import run_train_TDnCNN, run_finetune_TDnCNN
from filter_dataset import filter_dataset
from evaluation_pipe import epochs_evaluation

sys.path.append(os.getcwd()) 

# Инициализация Логгера
log = logging.getLogger(__name__)


def experiment_pipe(cfg: DictConfig):

    if cfg.task == 'preprocessing':
        # Препроцессинг train_dataset
        log.info("##### Starting preprocessing #####")
        log.info(OmegaConf.to_yaml(cfg.preprocessing))
        preprocessing_pipe(cfg.preprocessing)

    elif cfg.task == 'gen_noisy_dataset':
        # Симуляция зашумленных данных (осуществляется на train dataset)
        log.info("##### Starting gen_noisy_dataset #####")
        log.info(OmegaConf.to_yaml(cfg.gen_noisy_dataset))
        generate_noisy_dataset(cfg.gen_noisy_dataset)

    elif cfg.task == 'train_DDPM':
        # Обучение диффузионной модели
        log.info("##### Starting train_DDPM #####")
        log.info(OmegaConf.to_yaml(cfg.train_DDPM))
        run_train_DDPM(cfg.train_DDPM)

    elif cfg.task == 'finetune_DDPM':
        # Дообучение диффузионной модели
        log.info("##### Starting finetune_DDPM #####")
        log.info(OmegaConf.to_yaml(cfg.finetune_DDPM))
        run_finetune_DDPM(cfg.finetune_DDPM)

    elif cfg.task == 'train_TDnCNN':
        # Обучение денойзера
        log.info("##### Starting finetune_DDPM #####")
        log.info(OmegaConf.to_yaml(cfg.train_TDnCNN))
        run_train_TDnCNN(cfg.train_TDnCNN)

    elif cfg.task == 'finetune_TDnCNN':
        # Дообучение денозера
        log.info("##### Starting finetune_DDPM #####")
        log.info(OmegaConf.to_yaml(cfg.finetune_TDnCNN))
        run_finetune_TDnCNN(cfg.finetune_TDnCNN)

    elif cfg.task == 'filter_dataset':
        # Фильтрация датасета
        log.info("##### Starting filter_dataset #####")
        log.info(OmegaConf.to_yaml(cfg.filter_dataset))
        filter_dataset(cfg.filter_dataset)

    elif cfg.task == 'epochs_evaluation':
        # Расчет evaluation gap для множества эпох обучения денойзера
        log.info("##### Starting epochs_evaluation #####")
        log.info(OmegaConf.to_yaml(cfg.epochs_evaluation))
        epochs_evaluation(cfg.epochs_evaluation)
    


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    try:
        experiment_pipe(cfg)
    except Exception as e:
        log.error(f"Critical error in execution: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()