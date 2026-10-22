import random
from pathlib import Path

import comet_ml
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.logger.utils import plot_spectrogram
from src.text_encoder import TextEncoder
from src.utils.init_utils import set_random_seed


@hydra.main(
    version_base=None, config_path="src/configs", config_name="demo_augmentations"
)
def main(config):
    set_random_seed(config.seed)
    spectrogram = instantiate(config.spectrogram)
    augmentations = {
        name: instantiate(params) for name, params in config.augmentations.items()
    }
    dataset = instantiate(
        config.dataset,
        text_encoder=TextEncoder(),
        target_sr=config.sample_rate,
        instance_transforms={"get_spectrogram": spectrogram},
    )
    indices = random.Random(config.seed).sample(range(len(dataset)), config.n_samples)
    experiment = comet_ml.Experiment(
        project_name=config.comet.project,
        workspace=config.comet.workspace,
        auto_metric_logging=False,
        auto_param_logging=False,
        log_code=False,
    )

    experiment.set_name(config.experiment_name)
    experiment.log_parameters(OmegaConf.to_container(config, resolve=True))
    experiment.log_asset_data(
        OmegaConf.to_yaml(config, resolve=True), name="demo_config.yaml"
    )

    with torch.inference_mode():
        for step, index in enumerate(tqdm(indices, desc="Augmentations")):
            sample = dataset[index]
            audio = sample["original_audio"]
            audio_id = Path(sample["audio_path"]).stem
            experiment.log_text(
                sample["text"], step=step, metadata={"audio_id": audio_id}
            )
            versions = {"original": audio}
            versions.update(
                {
                    name: augmentation(audio.clone())
                    for name, augmentation in augmentations.items()
                }
            )
            specs = {
                name: spectrogram(wave).squeeze(0) for name, wave in versions.items()
            }
            for name, wave in versions.items():
                label = f"{audio_id}_{name}"
                experiment.log_audio(
                    audio_data=wave.numpy().T,
                    sample_rate=config.sample_rate,
                    file_name=f"{label}.wav",
                    step=step,
                )
                image = plot_spectrogram(specs[name], name=label)
                experiment.log_image(
                    image.permute(1, 2, 0).numpy(), name=label, step=step
                )

    experiment.end()


if __name__ == "__main__":
    main()
