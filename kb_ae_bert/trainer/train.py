import os
import sys
import logging
import pytorch_lightning as pl
from ..utils.config import *
from .kb_trainer import KBEncoderTrainer
from .qa_trainer import QATrainer
from .glue_trainer import GLUETrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


def find_checkpoint(config, stage_index):
    # temp for finding best checkpoint, only works if save k=1
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
    return os.path.join(checkpoint_dir, checkpoint), epoch


def train(config: Config, stage_index: int, only_test: bool = False):
    # execute pipeline
    is_distributed = len(config.gpus) > 1
    stage = config.pipeline[stage_index]
    stage_config = config.configs[stage_index]
    seed_everything(stage_config.seed, workers=True)

    checkpoint_path = os.path.join(
        config.working_directory, str(stage_index), "checkpoint"
    )
    log_path = os.path.join(config.working_directory, str(stage_index), "log")
    stage_result_path = os.path.join(
        config.working_directory, str(stage_index), "result"
    )

    if not only_test:
        logging.info("Training.")
        if stage not in ("kb_encoder", "qa", "glue"):
            raise ValueError(f"Unknown stage {stage}")

        if stage == "kb_encoder":
            stage_trainer = KBEncoderTrainer(
                stage_config, stage_result_path, is_distributed=is_distributed,
            )
        elif stage == "qa":
            stage_trainer = QATrainer(
                stage_config, stage_result_path, is_distributed=is_distributed
            )
        elif stage == "glue":
            stage_trainer = GLUETrainer(
                stage_config, stage_result_path, is_distributed=is_distributed
            )
        else:
            raise ValueError(f"Unknown stage {stage}.")

        # create directories, or reuse
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        save_config(config, os.path.join(config.working_directory, "config.json"))
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="{epoch:02d}-"
            + stage_trainer.monitor
            + "-{"
            + stage_trainer.monitor
            + ":.2f}",
            save_top_k=1 if stage_config.save else 0,
            save_last=False,
            monitor=stage_trainer.monitor,
            mode=stage_trainer.monitor_mode,
            verbose=True,
        )
        early_stopping = EarlyStopping(
            monitor=stage_trainer.monitor,
            mode=stage_trainer.monitor_mode,
            patience=config.early_stopping_patience,
            verbose=True,
        )
        t_logger = TensorBoardLogger(log_path)

        checkpoint = None
        if stage_config.load:
            checkpoint, _epoch = find_checkpoint(config, stage_index)
            if checkpoint is None:
                logging.info(
                    "Failed to find a valid checkpoint, using original weights."
                )
            else:
                logging.info(f"Using checkpoint {checkpoint}")
        else:
            logging.info("Not loading, using original weights.")

        trainer = pl.Trainer(
            gpus=config.gpus,
            accelerator="ddp" if len(config.gpus) > 1 else None,
            plugins=[DDPPlugin(find_unused_parameters=True)],
            callbacks=[checkpoint_callback, early_stopping],
            logger=[t_logger],
            limit_train_batches=getattr(stage_config, "train_steps", None) or 1.0,
            limit_val_batches=getattr(stage_config, "validate_steps", None) or 1.0,
            max_epochs=stage_config.epochs,
            # # For iterable datasets, to validate after each epoch,
            # # set check interval equal to number of training steps.
            # val_check_interval=stage_config.train_steps,
            accumulate_grad_batches=stage_config.accumulate_grad_batches,
            resume_from_checkpoint=checkpoint,
            deterministic=True,
        )

        trainer.fit(stage_trainer)
    else:
        logging.info("Testing.")

        if stage not in ("kb_encoder", "qa", "glue"):
            raise ValueError(f"Unknown stage {stage}")

        checkpoint, _epoch = find_checkpoint(config, stage_index)
        if checkpoint is None:
            raise RuntimeError("Cannot find a valid checkpoint for testing.")
        else:
            logging.info(f"Using checkpoint {checkpoint}")

        # model` must be provided to `trainer.test()` when it hasn't been passed
        # in a previous run.
        # the ckpt_path in test will be ignored in this case.
        # and must perform manual load
        if stage == "kb_encoder":
            stage_trainer = KBEncoderTrainer.load_from_checkpoint(
                checkpoint_path=checkpoint
            )
        elif stage == "qa":
            stage_trainer = QATrainer.load_from_checkpoint(checkpoint_path=checkpoint)
        elif stage == "glue":
            stage_trainer = GLUETrainer.load_from_checkpoint(checkpoint_path=checkpoint)
        else:
            raise ValueError(f"Unknown stage {stage}.")

        trainer = pl.Trainer(
            gpus=config.gpus,
            accelerator="ddp" if len(config.gpus) > 1 else None,
            plugins=[DDPPlugin(find_unused_parameters=True)],
            deterministic=True,
        )
        trainer.test(stage_trainer)

    if trainer.global_rank != 0:
        sys.exit(0)
