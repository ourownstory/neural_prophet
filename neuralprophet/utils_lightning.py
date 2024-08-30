import logging
import math
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.tuner.tuning import Tuner

from neuralprophet.configure import Train
from neuralprophet.logger import ProgressBar

log = logging.getLogger("NP.utils_lightning")


def smooth_loss_and_suggest(lr_finder, window=10):
    """
    Smooth loss using a Hamming filter.

    Parameters
    ----------
        loss : np.array
            Loss values

    Returns
    -------
        loss_smoothed : np.array
            Smoothed loss values
        lr: np.array
            Learning rate values
        suggested_lr: float
            Suggested learning rate based on gradient
    """
    lr_finder_results = lr_finder.results
    lr = lr_finder_results["lr"]
    loss = np.array(lr_finder_results["loss"])
    # Derive window size from num lr searches, ensure window is divisible by 2
    # half_window = math.ceil(round(len(loss) * 0.1) / 2)
    half_window = math.ceil(window / 2)
    # Pad sequence and initialialize hamming filter
    loss = np.pad(loss, pad_width=half_window, mode="edge")
    hamming_window = np.hamming(2 * half_window)
    # Convolve the over the loss distribution
    try:
        loss_smooth = np.convolve(
            hamming_window / hamming_window.sum(),
            loss,
            mode="valid",
        )[1:]
    except ValueError:
        log.warning(
            f"The number of loss values ({len(loss)}) is too small to apply smoothing with a the window size of "
            f"{window}."
        )

    # Suggest the lr with steepest negative gradient
    try:
        # Find the steepest gradient and the minimum loss after that
        suggestion_steepest = lr[np.argmin(np.gradient(loss_smooth))]
        suggestion_minimum = lr[np.argmin(np.array(lr_finder_results["loss"]))]
    except ValueError:
        log.error(
            f"The number of loss values ({len(loss)}) is too small to estimate a learning rate. Increase the number of "
            "samples or manually set the learning rate."
        )
        raise
    # get the tuner's default suggestion
    suggestion_default = lr_finder.suggestion(skip_begin=20, skip_end=10)

    log.info(f"Learning rate finder ---- default suggestion: {suggestion_default}")
    log.info(f"Learning rate finder ---- steepest: {suggestion_steepest}")
    log.info(f"Learning rate finder ---- minimum (not used): {suggestion_minimum}")
    if suggestion_steepest is not None and suggestion_default is not None:
        log_suggestion_smooth = np.log(suggestion_steepest)
        log_suggestion_default = np.log(suggestion_default)
        lr_suggestion = np.exp((log_suggestion_smooth + log_suggestion_default) / 2)
        log.info(f"Learning rate finder ---- log-avg: {lr_suggestion}")
    elif suggestion_steepest is None and suggestion_default is None:
        log.error("Automatic learning rate test failed. Please set manually the learning rate.")
        raise
    else:
        lr_suggestion = suggestion_steepest if suggestion_steepest is not None else suggestion_default

    log.info(f"Learning rate finder ---- returning: {lr_suggestion}")
    log.info(f"Learning rate finder ---- LR (start): {lr[:5]}")
    log.info(f"Learning rate finder ---- LR (end): {lr[-5:]}")
    log.info(f"Learning rate finder ---- LOSS (start): {loss[:5]}")
    log.info(f"Learning rate finder ---- LOSS (end): {loss[-5:]}")
    return loss, lr, lr_suggestion


def _smooth_loss(loss, beta=0.9):
    smoothed_loss = np.zeros_like(loss)
    smoothed_loss[0] = loss[0]
    for i in range(1, len(loss)):
        smoothed_loss[i] = smoothed_loss[i - 1] * beta + (1 - beta) * loss[i]
    return smoothed_loss


def configure_trainer(
    config_train: Train,
    metrics_logger,
    early_stopping_target: str = "Loss",
    accelerator: Optional[str] = None,
    progress_bar_enabled: bool = True,
    metrics_enabled: bool = False,
    checkpointing_enabled: bool = False,
    num_batches_per_epoch: int = 100,
    deterministic: bool = False,
):
    """
    Configures the PyTorch Lightning trainer.

    Parameters
    ----------
        config_train : Dict
            dictionary containing the overall training configuration.
        metrics_logger : MetricsLogger
            MetricsLogger object to log metrics to.
        early_stopping_target : str
            Target metric to use for early stopping.
        accelerator : str
            Accelerator to use for training.
        progress_bar_enabled : bool
            If False, no progress bar is shown.
        metrics_enabled : bool
            If False, no metrics are logged. Calculating metrics is computationally expensive and reduces the training
            speed.
        checkpointing_enabled : bool
            If False, no checkpointing is performed. Checkpointing reduces the training speed.
        num_batches_per_epoch : int
            Number of batches per epoch.

    Returns
    -------
        pl.Trainer
            PyTorch Lightning trainer
        checkpoint_callback
            PyTorch Lightning checkpoint callback to load the best model
    """
    if config_train.pl_trainer_config is None:
        config_train.pl_trainer_config = {}

    pl_trainer_config = config_train.pl_trainer_config
    # pl_trainer_config = pl_trainer_config.copy()

    # Set max number of epochs
    assert hasattr(config_train, "epochs") and config_train.epochs is not None
    pl_trainer_config["max_epochs"] = config_train.epochs

    # Configure the Ligthing-logs directory
    if "default_root_dir" not in pl_trainer_config.keys():
        pl_trainer_config["default_root_dir"] = os.getcwd()

    # Accelerator
    if isinstance(accelerator, str):
        if (accelerator == "auto" and torch.cuda.is_available()) or accelerator == "gpu":
            pl_trainer_config["accelerator"] = "gpu"
            pl_trainer_config["devices"] = -1
        elif (accelerator == "auto" and hasattr(torch.backends, "mps")) or accelerator == "mps":
            if torch.backends.mps.is_available():
                pl_trainer_config["accelerator"] = "mps"
                pl_trainer_config["devices"] = 1
        elif accelerator != "auto":
            pl_trainer_config["accelerator"] = accelerator
            pl_trainer_config["devices"] = 1

        if "accelerator" in pl_trainer_config:
            log.info(
                f"Using accelerator {pl_trainer_config['accelerator']} with {pl_trainer_config['devices']} device(s)."
            )
        else:
            log.info("No accelerator available. Using CPU for training.")

    # Configure metrics
    if metrics_enabled:
        pl_trainer_config["logger"] = metrics_logger
    else:
        pl_trainer_config["logger"] = False

    pl_trainer_config["deterministic"] = deterministic

    # Configure callbacks
    callbacks = []
    has_custom_callbacks = True if "callbacks" in pl_trainer_config else False

    # Configure checkpointing
    has_modelcheckpoint_callback = (
        True
        if has_custom_callbacks
        and any(isinstance(callback, pl.callbacks.ModelCheckpoint) for callback in pl_trainer_config["callbacks"])
        else False
    )
    if has_modelcheckpoint_callback and not checkpointing_enabled:
        raise ValueError(
            "Checkpointing is disabled but a ModelCheckpoint callback is provided. Please enable checkpointing or "
            "remove the callback."
        )
    if checkpointing_enabled:
        if not has_modelcheckpoint_callback:
            # Callback to access both the last and best model
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor=early_stopping_target, mode="min", save_top_k=1, save_last=True
            )
            callbacks.append(checkpoint_callback)
        else:
            checkpoint_callback = next(
                callback
                for callback in pl_trainer_config["callbacks"]
                if isinstance(callback, pl.callbacks.ModelCheckpoint)
            )
    else:
        pl_trainer_config["enable_checkpointing"] = False
        checkpoint_callback = None

    # Configure the progress bar, refresh every epoch
    has_progressbar_callback = (
        True
        if has_custom_callbacks
        and any(isinstance(callback, pl.callbacks.ProgressBar) for callback in pl_trainer_config["callbacks"])
        else False
    )
    if has_progressbar_callback and not progress_bar_enabled:
        raise ValueError(
            "Progress bar is disabled but a ProgressBar callback is provided. Please enable the progress bar or remove"
            " the callback."
        )
    if progress_bar_enabled:
        if not has_progressbar_callback:
            prog_bar_callback = ProgressBar(refresh_rate=num_batches_per_epoch, epochs=config_train.epochs)
            callbacks.append(prog_bar_callback)
    else:
        pl_trainer_config["enable_progress_bar"] = False

    # Early stopping monitor
    has_earlystopping_callback = (
        True
        if has_custom_callbacks
        and any(isinstance(callback, pl.callbacks.EarlyStopping) for callback in pl_trainer_config["callbacks"])
        else False
    )
    if has_earlystopping_callback and not config_train.early_stopping:
        raise ValueError(
            "Early stopping is disabled but an EarlyStopping callback is provided. Please enable early stopping or "
            "remove the callback."
        )
    if config_train.early_stopping:
        if not metrics_enabled:
            raise ValueError("Early stopping requires metrics to be enabled.")
        if not has_earlystopping_callback:
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor=early_stopping_target, mode="min", patience=20, divergence_threshold=5.0
            )
            callbacks.append(early_stop_callback)

    if has_custom_callbacks:
        pl_trainer_config["callbacks"].extend(callbacks)
    else:
        pl_trainer_config["callbacks"] = callbacks
    pl_trainer_config["num_sanity_val_steps"] = 0
    pl_trainer_config["enable_model_summary"] = False
    # TODO: Disabling sampler_ddp brings a good speedup in performance, however, check whether this is a good idea
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
    # config["replace_sampler_ddp"] = False

    return pl.Trainer(**pl_trainer_config), checkpoint_callback


def find_learning_rate(model, loader, trainer, train_epochs):
    log.info("No Learning Rate provided. Activating learning rate finder")

    # Configure the learning rate finder args
    batches_per_epoch = len(loader)
    main_training_total_steps = train_epochs * batches_per_epoch
    # main_training_total_steps is around 1e3 to 1e6 -> num_training 100 to 400
    num_training = 100 + int(np.log10(1 + main_training_total_steps / 1000) * 100)
    if batches_per_epoch < num_training:
        log.warning(
            f"Learning rate finder: The number of batches per epoch ({batches_per_epoch}) is too small than the required number \
                for the learning rate finder ({num_training}). The results might not be optimal."
        )
        # num_training = num_batches
    lr_finder_args = {
        "min_lr": 1e-7,
        "max_lr": 1e1,
        "num_training": num_training,
        "early_stop_threshold": None,
        "mode": "exponential",
    }
    log.info(f"Learning rate finder ---- ARGs: {lr_finder_args}")

    # Execute the learning rate range finder
    tuner = Tuner(trainer)
    model.finding_lr = True
    # model.train_loader = loader
    lr_finder = tuner.lr_find(
        model=model,
        train_dataloaders=loader,
        # val_dataloaders=val_loader, # not used, but lead to Lightning bug if not provided in prior versions.
        **lr_finder_args,
    )
    model.finding_lr = False

    # Estimate the optimal learning rate from the loss curve
    assert lr_finder is not None
    loss_list, lr_list, lr_suggested = smooth_loss_and_suggest(lr_finder)
    log.info(f"Learning rate finder suggested learning rate: {lr_suggested}")
    return lr_suggested
