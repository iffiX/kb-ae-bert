import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import AutoTokenizer, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
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
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed

        self.glue_tokenizer = AutoTokenizer.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )
        self.dataset = GLUEDataset(
            task=config.task,
            tokenizer=self.glue_tokenizer,
            kb_encoder_path=config.kb_encoder_path,
            kb_context_length=config.kb_encoder_context_length,
            kb_max_seq_length=config.kb_encoder_max_seq_length,
            kb_process_gpus=config.kb_process_gpus,
            kb_process_batch_size_per_gpu=config.kb_process_batch_size_per_gpu,
            storage_precision=config.storage_precision,
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

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.train_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset.validate_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.test_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
        )

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        extend_tokens = t.where(
            (batch["input_ids"] != self.glue_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.glue_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=batch["kb_embeds"].to(self.device),
            extend_tokens=extend_tokens.to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            labels=batch["label"].to(self.device),
        )
        return out[0]

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        extend_tokens = t.where(
            (batch["input_ids"] != self.glue_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.glue_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=batch["kb_embeds"].to(self.device),
            extend_tokens=extend_tokens.to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            labels=batch["label"].to(self.device),
        )
        batch = batch.to("cpu")
        return {
            "batch": BatchEncoding(
                {
                    "idx": batch["idx"],
                    "label": batch["label"],
                    "dataset_id": batch["dataset_id"],
                }
            ),
            "logits": out[1].cpu(),
        }

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        batch, logits = self.collate_and_filter_outputs(outputs)
        metrics = self.dataset.validate(batch, logits)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        if get_rank() == 0:
            print("Validation result:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

    def test_step(self, batch: BatchEncoding, _batch_idx):
        extend_tokens = t.where(
            (batch["input_ids"] != self.glue_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.glue_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.glue_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=batch["kb_embeds"].to(self.device),
            extend_tokens=extend_tokens.to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
        )
        batch = batch.to("cpu")
        return {
            "batch": BatchEncoding(
                {
                    "idx": batch["idx"],
                    "label": batch["label"],
                    "dataset_id": batch["dataset_id"],
                }
            ),
            "logits": out[1].cpu(),
        }

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_main_process(gathered_outputs)
        else:
            self.test_on_main_process(outputs)

    @rank_zero_only
    def test_on_main_process(self, outputs):
        _, logits = self.collate_and_filter_outputs(outputs)
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

    @staticmethod
    def collate_and_filter_outputs(outputs):
        batch = collate_function_dict_to_batch_encoding([o["batch"] for o in outputs])
        logits = t.cat([o["logits"] for o in outputs], dim=0)
        list_of_results = [
            (int(idx), int(d_id), lab.view(1), log.view(1, -1))
            for idx, d_id, lab, log in zip(
                batch["idx"], batch["dataset_id"], batch["label"], logits
            )
        ]
        # filter duplicates brought by resetting dataset
        existed = {}
        filtered = []
        for lr in list_of_results:
            if lr[0] not in existed:
                filtered.append(lr)
                existed[lr[0]] = True
        list_of_results = filtered
        list_of_results.sort(key=lambda lr: lr[0])
        logits = t.cat([lr[3] for lr in list_of_results], dim=0)
        batch = BatchEncoding(
            {
                "idx": t.tensor([lr[0] for lr in list_of_results]),
                "dataset_id": t.tensor([lr[1] for lr in list_of_results]),
                "label": t.cat([lr[2] for lr in list_of_results], dim=0).flatten(),
            }
        )
        return batch, logits
