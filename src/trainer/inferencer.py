from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.decoding.base import DecodeMethod
from src.metrics.text_metrics.base import TextMetricBase
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        logger,
        metrics,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        if not skip_model_load and config.inferencer.get("from_pretrained") is None:
            raise ValueError("Provide checkpoint or set skip_model_load=True")

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.decode_methods = [
            DecodeMethod(method) for method in self.cfg_trainer.decode_methods
        ]
        final_method = self.cfg_trainer.get("final_decode_method")
        self.final_decode_method = (
            None if final_method in (None, "none") else DecodeMethod(final_method)
        )
        if (
            self.final_decode_method is not None
            and self.final_decode_method not in self.decode_methods
        ):
            raise ValueError("final_decode_method must be included in decode_methods")

        self.device = device

        self.logger = logger

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        self.evaluation_metrics = MetricTracker(
            [m.name for m in self.metrics["inference"] if m.avg_type() == "mean"],
            sum_keys=[
                m.name for m in self.metrics["inference"] if m.avg_type() == "sum"
            ],
        )

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs

            for key, value in logs.items():
                self.logger.info("%s/%s: %.6f", part, key, value)

        return part_logs

    def process_batch(
        self,
        batch: dict,
        metrics: MetricTracker,
        part: str,
    ):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            processed_samples (int): cnt of processed samples.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]

        outputs = self.model.encode(**batch)
        batch.update(outputs)

        batch = self.decode_batch(batch)

        for met in metric_funcs:
            if (
                isinstance(met, TextMetricBase)
                and not self.evaluation_dataloaders[part].dataset.has_ground_truth()
            ):
                continue
            metrics.update_by_type(met.avg_type(), met.name, **met(**batch))

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()
        has_ground_truth = dataloader.dataset.has_ground_truth()

        self.evaluation_metrics.reset()

        audio_ids = []
        true_texts = []
        all_predictions = {
            str(decode_method): [] for decode_method in self.decode_methods
        }
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.inference_mode():
            for batch in tqdm(
                dataloader,
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

                for decode_method in self.decode_methods:
                    predictions = batch[f"text_predicted_{str(decode_method)}"]
                    all_predictions[str(decode_method)] += predictions

                audio_ids += list(
                    map(lambda audio_path: Path(audio_path).stem, batch["audio_path"])
                )

                if has_ground_truth:
                    true_texts += batch["text"]

        if self.save_path is not None:
            save_path = self.save_path / part / "predictions.csv"

            columns = {"id": audio_ids}
            if has_ground_truth:
                columns["true_text"] = true_texts
            pd.DataFrame(columns | all_predictions).to_csv(save_path, index=False)
            if self.final_decode_method is not None:
                transcription_dir = self.save_path / part / "transcriptions"
                transcription_dir.mkdir(parents=True, exist_ok=True)
                for audio_id, text in zip(
                    audio_ids,
                    all_predictions[str(self.final_decode_method)],
                    strict=True,
                ):
                    (transcription_dir / f"{audio_id}.txt").write_text(
                        text, encoding="utf-8"
                    )

        logs = self.evaluation_metrics.result()
        if not has_ground_truth:
            for metric in self.metrics["inference"]:
                if isinstance(metric, TextMetricBase):
                    logs.pop(metric.name, None)
            self.logger.info("%s: no reference transcriptions", part)
        return logs
