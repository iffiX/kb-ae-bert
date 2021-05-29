import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from ..model.kb_ae import KBMaskedLMEncoder
from ..model.ext_vocab import ExtendVocabForQA
from ..dataset.qa.squad import SQuADDataset
from ..dataset.kb.kdwd import KDWDDataset
from ..utils.config import Config, QATrainConfig, KBEncoderTrainConfig


class QATrainer(pl.LightningModule):
    def __init__(self, config: QATrainConfig):
        """
        Args:
            config:
        """
        super().__init__()
        np.random.seed(config.seed)
        t.random.manual_seed(config.seed)
        self.config = config

        self.qa_model = ExtendVocabForQA(
            base_type=config.base_type,
            extend_config=config.extend_config,
            extend_mode=config.extend_mode,
            **config.base_configs,
        )
        self.qa_tokenizer = AutoTokenizer.from_pretrained(config.base_type)
        if self.dataset_name == "squad":
            self.qa_dataset = SQuADDataset(tokenizer=self.qa_tokenizer)
        else:
            raise ValueError(
                f"Unknown QATrainConfig.dataset_name: {config.dataset_name}"
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.qa_dataset.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=lambda x: x,
        )

    def validation_dataloader(self):
        return DataLoader(
            dataset=self.qa_dataset.validate_dataset,
            batch_size=self.config.batch_size,
            collate_fn=lambda x: x,
        )

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, _batch_idx):
        pass

    def configure_optimizers(self):
        optim_cls = getattr(t.optim, self.config.optimizer_class)
        return optim_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization,
        )


def train(config: Config):
    # execute pipeline
    pass
