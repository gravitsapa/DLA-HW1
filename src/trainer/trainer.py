from pathlib import Path

import pandas as pd
import torch

from src.decoding.base import DecodeMethod
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(
        self,
        batch: dict,
        metrics: MetricTracker,
        now_is_eval: bool,
    ) -> dict:
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        if now_is_eval:
            for decode_method in DecodeMethod:
                decoded_tokens = self.model.decode(
                    decode_method=decode_method,
                    **batch,
                )
                batch.update(
                    {
                        f"{key}_{str(decode_method)}": val
                        for key, val in decoded_tokens.items()
                    }
                )
                prediction = self.text_encoder.decode_batch(
                    tokens=decoded_tokens["tokens_predicted"],
                    lengths=decoded_tokens["lengths_predicted"],
                )
                batch[f"text_predicted_{str(decode_method)}"] = prediction

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        loss_is_correct = bool(torch.isfinite(batch["loss"]).item())
        batch["loss_is_correct"] = loss_is_correct

        if self.is_train and loss_is_correct:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()

        if loss_is_correct:
            # update metrics for each loss (in case of multiple losses)
            for loss_name in self.config.writer.loss_names:
                metrics.update_mean(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update_by_type(met.avg_type(), met.name, **met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_original_audio(**batch)
            self.log_augmented_audio(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
            self.log_original_audio(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_audio(self, audio, audio_length, sample_rate, audio_path, name):
        index = 0
        length = int(audio_length[index].item())
        path = Path(audio_path[index]).name

        waveform = audio[index, :length].detach().cpu()

        self.writer.add_audio(
            f"({name}){path}",
            waveform,
            sample_rate=sample_rate,
        )

    def log_original_audio(
        self, original_audio, original_audio_length, sample_rate, audio_path, **batch
    ):
        self.log_audio(
            original_audio, original_audio_length, sample_rate, audio_path, "original"
        )

    def log_augmented_audio(
        self, audio, audio_length, sample_rate, audio_path, **batch
    ):
        self.log_audio(audio, audio_length, sample_rate, audio_path, "augmented")

    def log_predictions(self, text, audio_path, examples_to_log=10, **batch):
        for decode_method in DecodeMethod:
            texts_predicted = batch[f"text_predicted_{str(decode_method)}"]

            tuples = list(zip(texts_predicted, text, audio_path))

            rows = {}
            for pred, target, audio_path in tuples[:examples_to_log]:
                target = self.text_encoder.normalize_text(target)
                wer = calc_wer(target, pred) * 100
                cer = calc_cer(target, pred) * 100

                rows[Path(audio_path).name] = {
                    "step": self.writer.step,
                    "method": str(decode_method),
                    "target": target,
                    "predictions": pred,
                    "wer": wer,
                    "cer": cer,
                }
            self.writer.add_table(
                "predictions", pd.DataFrame.from_dict(rows, orient="index")
            )
