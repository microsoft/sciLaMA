"""Custom PyTorch Lightning callbacks for sciLaMA training."""
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .config import SciLaMAConfig


def first_epoch_beta_reached(config: SciLaMAConfig) -> int:
    """First epoch (0-indexed) when beta >= beta_end. Used for delayed checkpoint/early-stop start."""
    t = config.training
    if t.beta_warmup_rate <= 0:
        return 0
    n = (t.beta_end - t.beta_start) / t.beta_warmup_rate
    return int(np.ceil(t.epochs_before_beta_warmup + n))


def _checkpoint_best_filename_template(base_name: str, monitor: str) -> str:
    """Best checkpoints: {base}_best-{epoch}-{val_loss:.4f}.ckpt. Last: {base}_last.ckpt."""
    return f"{base_name}_best-{{epoch}}-{{{monitor}:.4f}}"


class DelayedModelCheckpoint(ModelCheckpoint):
    """Save only when current_epoch >= start_epoch. Best -> *_best-{epoch}-{val_loss}.ckpt; last -> *_last.ckpt."""

    def __init__(self, start_epoch: int, save_key: str, phase: str, save_top_k: int = 1, monitor: str = "val_loss", **kwargs):
        base_name = f"{save_key}_{phase}"
        kwargs.setdefault("save_last", True)
        kwargs["filename"] = _checkpoint_best_filename_template(base_name, monitor)
        kwargs["auto_insert_metric_name"] = True
        kwargs["save_top_k"] = save_top_k
        kwargs["monitor"] = monitor
        super().__init__(**kwargs)
        self._start_epoch = start_epoch
        self._base_name = base_name

    def _should_skip_save(self, trainer: pl.Trainer) -> bool:
        return trainer.current_epoch < self._start_epoch

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_skip_save(trainer):
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_skip_save(trainer):
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._should_skip_save(trainer):
            return
        super().on_train_end(trainer, pl_module)

    def _save_last_checkpoint(self, trainer: pl.Trainer, monitor_candidates: dict) -> None:
        if not self.save_last:
            return
        filepath = os.path.join(self.dirpath, f"{self._base_name}_last.ckpt")
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)


class DelayedEarlyStopping(EarlyStopping):
    """Only start counting patience from start_epoch (e.g. after beta reaches 1)."""

    def __init__(self, start_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self._start_epoch = start_epoch

    def _run_early_stopping_check(self, trainer: pl.Trainer) -> None:
        if trainer.current_epoch < self._start_epoch:
            return
        super()._run_early_stopping_check(trainer)


class PrintModelArchitecture(pl.Callback):
    """Print model architecture when training starts."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print("\n----- Model architecture -----")
        print(pl_module)
        print("-" * 50 + "\n")
