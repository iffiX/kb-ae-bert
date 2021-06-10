import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding
from ..model.kb_ae import KBMaskedLMEncoder
from ..dataset.base import collate_function_dict_to_batch_encoding
from ..dataset.kb.kdwd import KDWDDataset
from ..utils.config import KBEncoderTrainConfig
from ..utils.settings import proxies, model_cache_dir


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
        self.kb_tokenizer = AutoTokenizer.from_pretrained(
            config.base_type, cache_dir=model_cache_dir, proxies=proxies,
        )

        if config.dataset == "KDWD":
            self.dataset = KDWDDataset(
                relation_size=config.relation_size,
                context_length=config.context_length,
                sequence_length=self.kb_tokenizer.model_max_length,
                tokenizer=self.kb_tokenizer,
                **config.dataset_config.dict(),
            )
        else:
            raise ValueError(f"Unknown KBEncoderTrainConfig.dataset: {config.dataset}")
        if config.task not in ("entity", "relation"):
            raise ValueError(f"Unknown KBEncoderTrainConfig.task: {config.task}")

        # mongo client is not compatible with fork
        t.multiprocessing.set_start_method("spawn", force=True)

    @property
    def monitor(self):
        if self.config.task == "entity":
            return "mlm_loss"
        else:
            if self.config.dataset == "KDWD":
                return "total_loss"
            else:
                raise ValueError("Unknown dataset.")

    def train_dataloader(self):
        if self.config.task == "entity":
            return DataLoader(
                dataset=self.dataset.train_entity_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
            )
        else:
            return DataLoader(
                dataset=self.dataset.train_relation_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
            )

    def val_dataloader(self):
        if self.config.task == "entity":
            return DataLoader(
                dataset=self.dataset.validate_entity_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
            )
        else:
            return DataLoader(
                dataset=self.dataset.validate_relation_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
            )

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        batch.convert_to_tensors("pt")
        if self.config.task == "entity":
            # Masked Language model training
            out = self.kb_model(
                token_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            return out.loss
        else:
            # Relation encoding training
            # Make sure that your model is trained on MLM first
            relation_logits = self.kb_model.compute_relation(
                tokens1=batch["input_ids_1"].to(self.device),
                tokens2=batch["input_ids_2"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
            )
            result = self.dataset.get_loss(batch, relation_logits)
            return result[0] + result[1]

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        batch.convert_to_tensors("pt")
        if self.config.task == "entity":
            # Masked Language model validation
            out = self.kb_model(
                token_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            metrics = {"mlm_loss": out.loss}
        else:
            # Relation encoding training
            # Make sure that your model is trained on MLM first
            relation_logits = self.kb_model.compute_relation(
                tokens1=batch["input_ids_1"].to(self.device),
                tokens2=batch["input_ids_2"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
            )
            metrics = self.dataset.validate_relation_encode_dataset(relation_logits)

        for key, value in metrics.items():
            self.log(key, value)

    def configure_optimizers(self):
        optim_cls = getattr(t.optim, self.config.optimizer_class)
        return optim_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization,
        )
