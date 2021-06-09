import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding
from ..model.kb_ae import KBMaskedLMEncoder
from ..dataset.base import EmptyDataset
from ..dataset.kb.kdwd import KDWDDataset
from ..utils.config import KBEncoderTrainConfig


class KBEncoderTrainer(pl.LightningModule):
    def __init__(self, config: KBEncoderTrainConfig):
        super().__init__()
        np.random.seed(config.seed)
        t.random.manual_seed(config.seed)
        self.config = config
        self.kb_model = KBMaskedLMEncoder(
            relation_size=config.relation_size,
            base_type=config.base_type,
            relation_mode=config.relation_mode,
            mlp_hidden_size=config.mlp_hidden_size,
            **config.base_configs,
        )

        self.qa_tokenizer = AutoTokenizer.from_pretrained(config.base_type)

        if config.train_dataset_path is None:
            self.train_qa_dataset = None
        elif "squad" in config.train_dataset_path:
            self.train_qa_dataset = SQuADDataset(
                dataset_path=config.train_dataset_path, tokenizer=self.qa_tokenizer
            )
        else:
            raise ValueError(
                f"Unknown QATrainConfig.train_dataset_path: {config.train_dataset_path}"
            )

        if config.validate_dataset_path is None:
            self.validate_qa_dataset = None
        elif "squad" in config.validate_dataset_path:
            self.validate_qa_dataset = SQuADDataset(
                dataset_path=config.validate_dataset_path, tokenizer=self.qa_tokenizer
            )
        else:
            raise ValueError(
                f"Unknown QATrainConfig.validate_dataset_path: "
                f"{config.validate_dataset_path}"
            )

    def train_dataloader(self):
        if self.train_qa_dataset is not None:
            return DataLoader(
                dataset=self.train_qa_dataset.train_dataset,
                batch_size=self.config.batch_size,
            )
        else:
            return DataLoader(dataset=EmptyDataset())

    def validation_dataloader(self):
        if self.validate_qa_dataset is not None:
            return DataLoader(
                dataset=self.validate_qa_dataset.validate_dataset,
                batch_size=self.config.batch_size,
            )
        else:
            return DataLoader(dataset=EmptyDataset())

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        batch.convert_to_tensors("pt")
        if self.config.kb_encoder_trainable:
            self.kb_encoder.train()
        else:
            self.kb_encoder.eval()
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
        )
        extend_tokens = t.where(
            batch["input_ids"]
            != self.qa_tokenizer.cls_token_id & batch["input_ids"]
            != self.qa_tokenizer.sep_token_id & batch["input_ids"]
            != self.qa_tokenizer.pad_token_id,
            1,
            0,
        )
        out = self.qa_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            start_positions=batch["start_positions"].to(self.device),
            end_positions=batch["end_positions"].to(self.device),
        )
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        batch.convert_to_tensors("pt")
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
        )
        extend_tokens = t.where(
            batch["input_ids"]
            != self.qa_tokenizer.cls_token_id & batch["input_ids"]
            != self.qa_tokenizer.sep_token_id & batch["input_ids"]
            != self.qa_tokenizer.pad_token_id,
            1,
            0,
        )
        out = self.qa_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            start_positions=batch["start_positions"].to(self.device),
            end_positions=batch["end_positions"].to(self.device),
        )
        metrics = self.validate_qa_dataset.validate(batch, out[1], out[2])
        for key, value in metrics.items():
            self.log(key, value)

    def configure_optimizers(self):
        optim_cls = getattr(t.optim, self.config.optimizer_class)
        return optim_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization,
        )
