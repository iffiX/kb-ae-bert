import os
import pytorch_lightning as pl
from ..utils.config import *
from .kb_trainer import KBEncoderTrainer
from .qa_trainer import QATrainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def find_checkpoint(config, stage_ids):
    checkpoint_dir = os.path.join(
        config.working_directory, str(stage_ids), "checkpoint"
    )
    sorted_by_epoch = sorted(
        os.listdir(checkpoint_dir), key=lambda x: int(x.split("-")[0])
    )
    return os.path.join(checkpoint_dir, sorted_by_epoch[-1])


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
            else:
                stage_trainer = KBEncoderTrainer.load_from_checkpoint(
                    find_checkpoint(config, i)
                )
        elif stage == "qa":
            if not stage_config.load:
                kb_encoder = KBEncoderTrainer.load_from_checkpoint(
                    stage_config.kb_encoder_path
                ).kb_model
                stage_trainer = QATrainer(kb_encoder, stage_config)
            else:
                stage_trainer = QATrainer.load_from_checkpoint(
                    find_checkpoint(config, i)
                )
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
            accumulate_grad_batches=stage_config.accumulate_grad_batches,
        )
        trainer.fit(stage_trainer)
