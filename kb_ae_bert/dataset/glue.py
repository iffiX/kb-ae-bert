import os
import torch as t
import numpy as np

from datasets import (
    load_dataset,
    load_metric,
    concatenate_datasets,
    DownloadConfig,
    DatasetDict,
)
from transformers import PreTrainedTokenizerBase, BatchEncoding
from kb_ae_bert.utils.file import open_file_with_create_directories
from kb_ae_bert.utils.settings import (
    dataset_cache_dir,
    metrics_cache_dir,
    preprocess_cache_dir,
    proxies,
)


class GLUEDataset:
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    task_to_datasets = {
        "cola": ("cola",),
        "mnli": ("mnli", "ax"),
        "mrpc": ("mrpc",),
        "qnli": ("qnli",),
        "qqp": ("qqp",),
        "rte": ("rte",),
        "sst2": ("sst2",),
        "stsb": ("stsb",),
        "wnli": ("wnli",),
    }

    task_to_reports = {
        "cola": ("CoLA.tsv",),
        "mnli": ("MNLI-m.tsv", "MNLI-mm.tsv", "AX.tsv"),
        "mrpc": ("MRPC.tsv",),
        "qnli": ("QNLI.tsv",),
        "qqp": ("QQP.tsv",),
        "rte": ("RTE.tsv",),
        "sst2": ("SST-2.tsv",),
        "stsb": ("STS-B.tsv",),
        "wnli": ("WNLI.tsv",),
    }

    def __init__(
        self,
        task: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 128,
        max_train_samples: int = None,
        max_validate_samples: int = None,
        max_test_samples: int = None,
    ):
        if task not in self.task_to_keys:
            raise ValueError(
                f"Invalid task '{task}', valid ones are {self.task_to_keys.keys()}"
            )
        if max_seq_length > tokenizer.model_max_length:
            raise ValueError(
                f"Max sequence length {max_seq_length} is larger than "
                f"max allowed length {tokenizer.model_max_length}"
            )
        huggingface_path = str(os.path.join(dataset_cache_dir, "huggingface"))

        self.task = task
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_train_samples = max_train_samples
        self.max_validate_samples = max_validate_samples
        self.max_test_samples = max_test_samples
        self.datasets = [
            load_dataset(
                path="glue",
                name=task,
                cache_dir=huggingface_path,
                download_config=DownloadConfig(proxies=proxies),
            )
            for task in self.task_to_datasets[task]
        ]

        self.metric = load_metric(
            "glue",
            config_name=task,
            cache_dir=metrics_cache_dir,
            download_config=DownloadConfig(proxies=proxies),
        )
        self.is_regression = self.task == "stsb"
        self._train = None
        self._validate = None
        self._test = None
        self.preprocess()
        if self.is_regression:
            self.num_labels = 1
        else:
            self.num_labels = self._train.features["label"].num_classes

    @property
    def train_dataset(self):
        return self._train

    @property
    def validate_dataset(self):
        return self._validate

    @property
    def test_dataset(self):
        return self._test

    @property
    def validate_size(self):
        return len(self._validate)

    @property
    def test_size(self):
        return len(self._test)

    def validate(self, batch: BatchEncoding, logits: t.Tensor):
        logits = logits.cpu().numpy()
        labels = np.squeeze(logits) if self.is_regression else np.argmax(logits, axis=1)
        ref_labels = batch["label"].cpu().numpy()
        if self.task != "mnli":
            return self.metric.compute(predictions=labels, references=ref_labels)
        else:
            mnli_m_idx = [
                idx
                for idx in range(len(labels))
                if batch["dataset"][idx] == "mnli" and "matched" in batch["sub-dataset"]
            ]
            mnli_mm_idx = [
                idx
                for idx in range(len(labels))
                if batch["dataset"][idx] == "mnli"
                and "mismatched" in batch["sub-dataset"]
            ]
            mnli_m_metric = self.metric.compute(
                predictions=labels[mnli_m_idx], references=ref_labels[mnli_m_idx]
            )
            mnli_mm_metric = self.metric.compute(
                predictions=labels[mnli_mm_idx], references=ref_labels[mnli_mm_idx]
            )
            return {
                "accuracy": (mnli_m_metric["accuracy"] + mnli_mm_metric["accuracy"])
                / 2,
                "mnli_matched_accuracy": mnli_m_metric["accuracy"],
                "mnli_mismatched_accuracy": mnli_mm_metric["accuracy"],
            }

    def generate_test_results(self, logits: t.Tensor, directory: str):
        logits = logits.cpu().numpy()
        labels = np.squeeze(logits) if self.is_regression else np.argmax(logits, axis=1)

        if len(labels) != len(self._test):
            raise ValueError(
                f"Total test number {len(self._test)}, "
                f"but input number is {len(labels)}"
            )

        # File format is specified by https://gluebenchmark.com/faq FAQ #1
        if self.is_regression:
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][0]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels):
                    file.write(f"{index}\t{item:3.3f}\n")
        elif self.task != "mnli":
            label_list = self._test.features["label"].names
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][0]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels):
                    item = label_list[item]
                    file.write(f"{index}\t{item}\n")
        else:
            mnli_m_idx = [
                idx
                for idx in range(len(labels))
                if self._test["dataset"][idx] == "mnli"
                and "matched" in self._test["sub-dataset"]
            ]
            mnli_mm_idx = [
                idx
                for idx in range(len(labels))
                if self._test["dataset"][idx] == "mnli"
                and "mismatched" in self._test["sub-dataset"]
            ]
            ax_idx = [
                idx for idx in range(len(labels)) if self._test["dataset"][idx] == "ax"
            ]
            # matched
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][0]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[mnli_m_idx]):
                    file.write(f"{index}\t{item}\n")

            # mismatched
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][1]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[mnli_mm_idx]):
                    file.write(f"{index}\t{item}\n")

            # ax
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][2]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[ax_idx]):
                    file.write(f"{index}\t{item}\n")

    def preprocess(self):
        # Pre-processing the raw_datasets
        sentence1_key, sentence2_key = self.task_to_keys[self.task]

        processed_datasets = []
        for dataset, dataset_name in zip(
            self.datasets, self.task_to_datasets[self.task]
        ):
            new_dataset = {}
            for sub_dataset_name in dataset:

                def preprocess_function(examples):
                    # Tokenize the texts
                    args = (
                        (examples[sentence1_key],)
                        if sentence2_key is None
                        else (examples[sentence1_key], examples[sentence2_key])
                    )
                    encodings = self.tokenizer(
                        *args,
                        padding="max_length",
                        max_length=self.max_seq_length,
                        truncation=True,
                    )
                    if self.task == "mnli":
                        encodings.update(
                            {
                                "dataset": ["dataset_name"] * len(examples),
                                "sub-dataset": sub_dataset_name,
                            }
                        )
                    return encodings

                os.makedirs(
                    os.path.join(preprocess_cache_dir, "glue_cache"), exist_ok=True
                )
                new_dataset[sub_dataset_name] = dataset[sub_dataset_name].map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=True,
                    cache_file_name=str(
                        os.path.join(
                            preprocess_cache_dir,
                            "glue_cache",
                            f"glue_{self.task}_{dataset_name}_{sub_dataset_name}.cache",
                        )
                    ),
                )

            new_dataset = DatasetDict(new_dataset)
            processed_datasets.append(new_dataset)

        preprocess_task_function = getattr(
            self, f"preprocess_task_{self.task}", self.preprocess_task_general
        )
        preprocess_task_function(processed_datasets)

    def preprocess_task_general(self, processed_datasets):
        self._train = processed_datasets[0]["train"]
        self._train = self.safe_select_samples(self._train, self.max_train_samples)

        self._validate = processed_datasets[0]["validation"]
        self._validate = self.safe_select_samples(
            self._validate, self.max_validate_samples
        )

        self._test = processed_datasets[0]["test"]
        self._test = self.safe_select_samples(self._test, self.max_test_samples)

    def preprocess_task_mnli(self, processed_datasets):
        self._train = processed_datasets[0]["train"]
        self._train = self.safe_select_samples(self._train, self.max_train_samples)

        self._validate = concatenate_datasets(
            [
                self.safe_select_samples(
                    processed_datasets[0]["validation_matched"],
                    self.max_validate_samples // 2,
                ),  # mnli
                self.safe_select_samples(
                    processed_datasets[0]["validation_mismatched"],
                    self.max_validate_samples - self.max_validate_samples // 2,
                ),  # mnli
            ]
        )

        self._test = concatenate_datasets(
            [
                self.safe_select_samples(
                    processed_datasets[0]["test_matched"], self.max_test_samples // 3
                ),  # mnli
                self.safe_select_samples(
                    processed_datasets[0]["test_mismatched"], self.max_test_samples // 3
                ),  # mnli
                self.safe_select_samples(
                    processed_datasets[1]["test"],
                    self.max_test_samples - self.max_test_samples // 3,
                ),  # ax
            ]
        )

    @staticmethod
    def safe_select_samples(dataset, max_number=None):
        if max_number is not None:
            if max_number <= 0:
                raise ValueError(
                    f"Select number must be greater than 0, but got {max_number}"
                )
            max_number = min(len(dataset), max_number)
            dataset = dataset.select(range(max_number))
        return dataset
