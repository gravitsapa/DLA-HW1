from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        if torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    now_is_eval=False,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            if batch["loss_is_correct"]:
                self.train_metrics.update_mean("grad_norm", self._get_grad_norm())

            # log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = last_train_metrics
        self.writer.set_step(epoch * self.epoch_len)
        logs.update(self._log_peak_gpu_memory())

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        if torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                    now_is_eval=True,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        logs = self.evaluation_metrics.result()
        self.writer.set_step(epoch * self.epoch_len, part)
        logs.update(self._log_peak_gpu_memory())
        return logs

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
            batch = self.decode_batch(batch)

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
        index = torch.randint(len(batch["audio_path"]), ()).item()
        self.log_spectrogram(index=index, **batch)
        self.log_original_audio(index=index, **batch)

        if mode == "train":  # the method is called only every self.log_step steps
            self.log_augmented_audio(index=index, **batch)
        else:
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, spectrogram_length, index, **batch):
        length = int(spectrogram_length[index].item())
        spectrogram_for_plot = spectrogram[index, :, :length].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_audio(self, audio, audio_length, sample_rate, audio_path, name, index):
        length = int(audio_length[index].item())
        path = Path(audio_path[index]).name

        waveform = audio[index, :length].detach().cpu()

        self.writer.add_audio(
            f"({name}){path}",
            waveform,
            sample_rate=sample_rate,
        )

    def log_original_audio(
        self,
        original_audio,
        original_audio_length,
        sample_rate,
        audio_path,
        index,
        **batch,
    ):
        self.log_audio(
            original_audio,
            original_audio_length,
            sample_rate,
            audio_path,
            "original",
            index=index,
        )

    def log_augmented_audio(
        self, audio, audio_length, sample_rate, audio_path, index, **batch
    ):
        self.log_audio(
            audio, audio_length, sample_rate, audio_path, "augmented", index=index
        )

    def log_predictions(self, text, audio_path, examples_to_log=10, **batch):
        for decode_method in self.decode_methods:
            if f"text_predicted_{str(decode_method)}" not in batch:
                continue
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
