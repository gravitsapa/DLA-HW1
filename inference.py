import json
import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import (
    log_git_commit_and_patch,
    set_random_seed,
    setup_logging,
)
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    save_dir = ROOT_PATH / config.inferencer.save_dir / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(save_dir)
    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)

    config_path = save_dir / "config.yaml"
    OmegaConf.save(config=config, f=config_path, resolve=True)
    log_git_commit_and_patch(save_dir)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(
        config.model,
        n_tokens=len(text_encoder),
        blank_id=text_encoder.blank_id,
    ).to(device)
    logger.info(model)

    # get metrics
    metrics = {"inference": []}
    for metric_type in ["inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metric = instantiate(metric_config, text_encoder=text_encoder)
            if metric.inference_only() and metric_type != "inference":
                raise ValueError(f"Metirc {metric.name} can only be in train mode")
            metrics[metric_type].append(metric)

    # save_path for model predictions
    if config.inferencer.save_predictions:
        save_path = save_dir / "predictions"
        save_path.mkdir(exist_ok=True, parents=True)
    else:
        save_path = None

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        logger=logger,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(logs, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
