###
# Author: Kai Li
# Date: 2022-05-26 18:09:54
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 16:00:58
###
import random
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from .. import losses
from ..utils.chunk_mask import sample_chunk_config


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioVisualLightningModuleAE(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        optimizer=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Streaming/chunk-aware training config
        self.streaming_config = self.config.get("streaming", {})
        self.streaming_enabled = self.streaming_config.get("enabled", False)

        # print("device:",self.device)
        self.loss_func = {
            "train": getattr(losses, config["loss"]["train"]["loss_func"])(
                getattr(losses, config["loss"]["train"]["sdr_type"])(**config["loss"]["train"]["stftconfig"],device=self.device),
                **config["loss"]["train"]["config"],
            ),
            "val": getattr(losses, config["loss"]["val"]["loss_func"])(
                getattr(losses, config["loss"]["val"]["sdr_type"]),
                **config["loss"]["val"]["config"],
            ),
        }

    def forward(self, wav, mouth=None, key=None,
                chunk_size=None, history_len=None, future_len=None):
        """Applies forward pass of the model.

        Args:
            wav: audio waveform
            mouth: video mouth ROI
            key: optional key
            chunk_size: streaming chunk size in waveform samples (or None)
            history_len: streaming history context in samples (or None)
            future_len: streaming future lookahead in samples (or None)

        Returns:
            :class:`torch.Tensor`
        """
        wav = wav.float()
        if mouth is not None and mouth.ndim == 4:
            mouth = mouth.unsqueeze(1)
        if mouth is not None:
            mouth = mouth.to(dtype=wav.dtype)
        return self.audio_model(wav, mouth,
                                chunk_size=chunk_size,
                                history_len=history_len,
                                future_len=future_len)

    def training_step(self, batch, batch_nb):
        mixtures, targets, mouth, key = batch

        # Sample streaming chunk params if enabled
        chunk_size = None
        history_len = None
        future_len = None
        if self.streaming_enabled and self._should_apply_chunk_mask():
            chunk_cfg = sample_chunk_config(
                self.streaming_config,
                audio_encoder_stride=self.audio_model.audio_encoder_stride,
                num_separator_stages=self.audio_model.separator.num_stages,
            )
            chunk_size = chunk_cfg.chunk_size
            history_len = chunk_cfg.history_len
            future_len = chunk_cfg.future_len

        model_out = self(mixtures, mouth, key,
                         chunk_size=chunk_size,
                         history_len=history_len,
                         future_len=future_len)
        if isinstance(model_out, (tuple, list)):
            est_sources, est_sources_bn = model_out[:2]
        else:
            est_sources = model_out
            est_sources_bn = model_out
        if targets.ndim == 2:
            targets = targets.unsqueeze(1)

        loss = self.loss_func["train"](est_sources, est_sources_bn, targets, self.current_epoch)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb, dataloader_idx):
        # cal val loss
        if dataloader_idx == 0:
            mixtures, targets, mouth, key = batch

            model_out = self(mixtures, mouth, key)
            if isinstance(model_out, (tuple, list)):
                est_sources, est_sources_bn = model_out[:2]
            else:
                est_sources = model_out
                est_sources_bn = model_out
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            loss = self.loss_func["val"](est_sources, targets)
            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            
            self.validation_step_outputs.append(loss)
            
            return {"val_loss": loss}

        # cal test loss
        if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
            mixtures, targets, mouth, key = batch
            model_out = self(mixtures, mouth, key)
            if isinstance(model_out, (tuple, list)):
                est_sources, est_sources_bn = model_out[:2]
            else:
                est_sources = model_out
                est_sources_bn = model_out
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            tloss = self.loss_func["val"](est_sources, targets)
            self.log(
                "test_loss",
                tloss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            self.test_step_outputs.append(tloss)
            return {"test_loss": tloss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.logger.experiment.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"], "epoch": self.current_epoch}
        )
        self.logger.experiment.log(
            {"val_pit_sisnr": -val_loss, "epoch": self.current_epoch}
        )

        # test
        if (self.trainer.current_epoch) % 10 == 0:
            avg_loss = torch.stack(self.test_step_outputs).mean()
            test_loss = torch.mean(self.all_gather(avg_loss))
            self.logger.experiment.log(
                {"test_pit_sisnr": -test_loss, "epoch": self.current_epoch}
            )
        self.validation_step_outputs.clear()  # free memory
        self.test_step_outputs.clear()  # free memory

    def _should_apply_chunk_mask(self) -> bool:
        """Decide whether to apply chunk masking for this training step.

        Supports a gradual unmasking schedule:
        - warmup_epochs: no masking (full context)
        - ramp_epochs: gradually increase masking probability
        - after ramp: always mask
        """
        schedule = self.streaming_config.get("schedule", {})
        if not schedule.get("enabled", False):
            return True

        warmup = schedule.get("warmup_epochs", 10)
        ramp = schedule.get("ramp_epochs", 20)
        epoch = self.current_epoch

        if epoch < warmup:
            return False
        elif epoch < warmup + ramp:
            prob = (epoch - warmup) / ramp
            return random.random() < prob
        else:
            return True

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                if 'residual' in v:
                    mapping = {
                        'residual': 0,
                        'compress_space': 1,
                        'consecutive_residual': 2,
                        'attend_space':3,
                        'linear_attend_space':4
                    }
                    v = [mapping[s] for s in v]
                dic[k] = torch.tensor(v)
        return dic