import os
import logging
import pytorch_lightning as pl
from ..utils.config import *
from .kb_trainer import KBEncoderTrainer
from .qa_trainer import QATrainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def find_checkpoint(config, stage_index):
    checkpoint_dir = os.path.join(
        config.working_directory, str(stage_index), "checkpoint"
    )
    sorted_by_epoch = sorted(
        os.listdir(checkpoint_dir), key=lambda x: int(x.split("-")[0].strip("epoch="))
    )
    if len(sorted_by_epoch) == 0:
        return None, None
    checkpoint = sorted_by_epoch[-1]
    epoch = int(checkpoint.split("-")[0].strip("epoch="))
    logging.info(f"Using checkpoint {checkpoint}")
    return os.path.join(checkpoint_dir, checkpoint), epoch


def safe_load(trainer_class, stage_index, config):
    checkpoint, version = find_checkpoint(config, stage_index)
    if checkpoint is None:
        logging.warning(
            f"Checkpoint not found for trainer {trainer_class}, "
            f"stage_index={stage_index}, restart from beginning."
        )
        # epoch is 0
        return trainer_class(config=config.configs[stage_index]), 0
    try:
        stage_trainer = trainer_class.load_from_checkpoint(checkpoint)
    except TypeError:
        # use current config
        logging.warning(
            "Note: config(hparams) not found in checkpoint, "
            "using current config and continue."
        )
        stage_trainer = trainer_class.load_from_checkpoint(
            checkpoint, config=config.configs[stage_index]
        )
    # reset epoch number to version + 1
    return stage_trainer, version + 1


def train(config: Config):
    # execute pipeline
    for i, (stage, stage_config) in enumerate(zip(config.pipeline, config.configs)):
        checkpoint_path = os.path.join(config.working_directory, str(i), "checkpoint")
        log_path = os.path.join(config.working_directory, str(i), "log")

        if stage not in ("kb_encoder", "qa"):
            raise ValueError(f"Unknown stage {stage}")

        if stage == "kb_encoder":
            if not stage_config.load:
                stage_trainer = KBEncoderTrainer(stage_config)
                epoch = 0
            else:
                stage_trainer, epoch = safe_load(KBEncoderTrainer, i, config)
        elif stage == "qa":
            if not stage_config.load:
                stage_trainer = QATrainer(stage_config)
                epoch = 0
            else:
                stage_trainer, epoch = safe_load(QATrainer, i, config)
        else:
            raise ValueError(f"Unknown stage {stage}.")

        # create directories, or reuse
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="{epoch:02d}-"
            + stage_trainer.monitor
            + "-{"
            + stage_trainer.monitor
            + ":.2f}",
            save_top_k=1,
            monitor=stage_trainer.monitor,
            mode="min",
            period=1,
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor=stage_trainer.monitor,
            mode="min",
            patience=config.early_stopping_patience,
        )
        t_logger = TensorBoardLogger(log_path)

        trainer = pl.Trainer(
            gpus=config.gpus,
            callbacks=[checkpoint_callback, early_stopping],
            logger=[t_logger],
            limit_train_batches=stage_config.train_steps,
            limit_val_batches=stage_config.validate_steps,
            max_epochs=stage_config.epochs,
            # For iterable datasets, to validate after each epoch,
            # set check interval equal to number of training steps.
            val_check_interval=stage_config.train_steps,
            accumulate_grad_batches=stage_config.accumulate_grad_batches,
        )
        trainer.current_epoch = epoch
        trainer.fit(stage_trainer)
