import os
import itertools
import numpy as np
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import gather_object, get_rank, get_world_size
from transformers import AutoTokenizer, BatchEncoding
from .kb_trainer import KBEncoderTrainer
from ..model.ext_vocab import ExtendVocabForSequenceClassification
from ..dataset.base import collate_function_dict_to_batch_encoding
from kb_ae_bert.dataset.glue import GLUEDataset
from ..utils.config import GLUETrainConfig
from ..utils.settings import proxies, model_cache_dir, huggingface_mirror


class GLUETrainer(pl.LightningModule):
    def __init__(
        self, config: GLUETrainConfig, stage_result_path="./", is_distributed=False
    ):
        super().__init__()
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        self.save_hyperparameters()

        np.random.seed(config.seed)
        t.random.manual_seed(config.seed)
        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed

        self.kb_encoder = KBEncoderTrainer.load_from_checkpoint(
            config.kb_encoder_path, only_init_model=True
        ).kb_model

        self.glue_tokenizer = AutoTokenizer.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )
        self.dataset = GLUEDataset(
            task=config.task,
            tokenizer=self.glue_tokenizer,
            max_seq_length=config.max_seq_length,
            max_train_samples=config.max_train_samples,
            max_validate_samples=config.max_validate_samples,
            max_test_samples=config.max_test_samples,
        )

        self.glue_model = ExtendVocabForSequenceClassification(
            base_type=config.base_type,
            extend_config=config.extend_config,
            extend_mode=config.extend_mode,
            num_labels=self.dataset.num_labels,
            **config.base_configs,
        )

    @property
    def monitor(self):
        task_to_monitor = {
            "cola": "matthews_correlation",
            "mnli": "accuracy",
            "mrpc": "f1",
            "qnli": "accuracy",
            "qqp": "f1",
            "rte": "accuracy",
            "sst2": "accuracy",
            "stsb": "pearson",
            "wnli": "accuracy",
        }
        return task_to_monitor[self.config.task]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset.validate_dataset,
            batch_size=1,
            collate_fn=collate_function_dict_to_batch_encoding,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.test_dataset,
            batch_size=1,
            collate_fn=collate_function_dict_to_batch_encoding,
        )

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        with_gradient_num = (
            0
            if not self.config.kb_encoder_trainable
            else self.config.kb_encoder_with_gradient_num
        )
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
            with_gradient_num=with_gradient_num,
        )
        extend_tokens = t.where(
            (batch["input_ids"] != self.glue_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.glue_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            labels=batch["label"].to(self.device),
        )
        return out[0]

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
            with_gradient_num=0,
        )
        extend_tokens = t.where(
            (batch["input_ids"] != self.glue_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.glue_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            labels=batch["label"].to(self.device),
        )
        return {"batch": batch.to("cpu"), "logits": out[1].cpu()}

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            gathered_outputs = [None] * get_world_size() if get_rank() == 0 else None
            gather_object(outputs, object_gather_list=gathered_outputs, dst=0)
            if gathered_outputs is not None:
                gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
                self.validate_on_main_process(gathered_outputs)
        else:
            self.validate_on_main_process(outputs)

    def validate_on_main_process(self, outputs):
        batch = collate_function_dict_to_batch_encoding([o["batch"] for o in outputs])
        logits = t.cat([o["logits"] for o in outputs], dim=0)
        metrics = self.dataset.validate(batch, logits)
        for key, value in metrics.items():
            self.log(key, value)

    def test_step(self, batch: BatchEncoding, _batch_idx):
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
            with_gradient_num=0,
        )
        extend_tokens = t.where(
            (batch["input_ids"] != self.glue_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.glue_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
        )
        return {"batch": batch.to("cpu"), "logits": out[1].cpu()}

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            gathered_outputs = [None] * get_world_size() if get_rank() == 0 else None
            gather_object(outputs, object_gather_list=gathered_outputs, dst=0)
            if gathered_outputs is not None:
                gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
                self.test_on_main_process(gathered_outputs)
        else:
            self.test_on_main_process(outputs)

    def test_on_main_process(self, outputs):
        batch = collate_function_dict_to_batch_encoding([o["batch"] for o in outputs])
        list_of_logits = [(idx, o["logits"]) for idx, o in zip(batch["idx"], outputs)]
        list_of_logits.sort(key=lambda l: l[0])
        logits = t.cat([ll[1] for ll in list_of_logits], dim=0)
        assert logits.shape[0] == self.dataset.test_size, (
            f"Size not match, input is {logits.shape[0]}, "
            f"reference is {self.dataset.test_size}"
        )

        self.dataset.generate_test_results(logits, self.stage_result_path)

    def configure_optimizers(self):
        optim_cls = getattr(t.optim, self.config.optimizer_class)
        return optim_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization,
        )
