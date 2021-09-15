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


def train(config: Config):
    # execute pipeline
    is_distributed = len(config.gpus) > 1
    for stage_index, (stage, stage_config) in enumerate(
        zip(config.pipeline, config.configs)
    ):
        checkpoint_path = os.path.join(
            config.working_directory, str(stage_index), "checkpoint"
        )
        log_path = os.path.join(config.working_directory, str(stage_index), "log")

        if stage not in ("kb_encoder", "qa"):
            raise ValueError(f"Unknown stage {stage}")

        if stage == "kb_encoder":
            stage_trainer = KBEncoderTrainer(
                stage_config, is_distributed=is_distributed
            )
        elif stage == "qa":
            stage_trainer = QATrainer(stage_config, is_distributed=is_distributed)
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
            save_top_k=2,
            save_last=True,
            monitor=stage_trainer.monitor,
            mode="min",
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor=stage_trainer.monitor,
            mode="min",
            patience=config.early_stopping_patience,
            verbose=True,
        )
        t_logger = TensorBoardLogger(log_path)

        checkpoint = None
        if stage_config.load:
            checkpoint, _version = find_checkpoint(config, stage_index)

        trainer = pl.Trainer(
            gpus=config.gpus,
            accelerator="ddp" if len(config.gpus) > 0 else None,
            callbacks=[checkpoint_callback, early_stopping],
            logger=[t_logger],
            limit_train_batches=stage_config.train_steps or 1.0,
            limit_val_batches=stage_config.validate_steps or 1.0,
            max_epochs=stage_config.epochs,
            # # For iterable datasets, to validate after each epoch,
            # # set check interval equal to number of training steps.
            # val_check_interval=stage_config.train_steps,
            accumulate_grad_batches=stage_config.accumulate_grad_batches,
            resume_from_checkpoint=checkpoint,
        )
        trainer.fit(stage_trainer)
