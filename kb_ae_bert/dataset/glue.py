import os
import h5py
import torch as t
import numpy as np
import logging
from typing import List
from tqdm import tqdm
from datasets import (
    load_dataset,
    load_metric,
    DownloadConfig,
)
from transformers import PreTrainedTokenizerBase, BatchEncoding
from kb_ae_bert.model.kb_ae import KBMaskedLMEncoder
from kb_ae_bert.utils.file import open_file_with_create_directories
from kb_ae_bert.utils.settings import (
    dataset_cache_dir,
    metrics_cache_dir,
    preprocess_cache_dir,
    proxies,
)
from .base import StaticIterableDataset


class KBProxy(t.nn.Module):
    def __init__(self, kb_encoder):
        super().__init__()
        self.kb_encoder = kb_encoder

    def forward(self, *args, **kwargs):
        return self.kb_encoder.compute_sentence_embeds(*args, **kwargs)


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

    AX = 0
    MNLI_MATCHED = 1
    MNLI_MISMATCHED = 2

    def __init__(
        self,
        task: str,
        tokenizer: PreTrainedTokenizerBase,
        kb_model: KBMaskedLMEncoder,
        kb_context_length: int,
        kb_max_seq_length: int,
        kb_process_gpus: List[int],
        kb_process_batch_size_per_gpu: int = 32,
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
        self.kb_context_length = kb_context_length
        self.kb_max_seq_length = kb_max_seq_length

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
        if self.is_regression:
            self.num_labels = 1
        else:
            self.num_labels = self.datasets[0]["train"].features["label"].num_classes

        self.file = None
        data_path = os.path.join(preprocess_cache_dir, "glue_data", f"{self.task}.hdf5")
        if os.path.exists(data_path):
            try:
                with h5py.File(data_path, "r",) as file:
                    assert len(file[self.task]["train"]["idx"]) > 1
            except Exception as e:
                logging.info(
                    f"Exception [{str(e)}] occurred while reading data file, "
                    f"regenerating data."
                )
                os.remove(data_path)
                logging.info(f"Data for GLUE[{self.task}] not found, generating.")
                self.preprocess(
                    kb_model, kb_process_gpus, kb_process_batch_size_per_gpu
                )
            else:
                logging.info(f"Found data for GLUE[{self.task}], skipping generation.")
        else:
            logging.info(f"Data for GLUE[{self.task}] not found, generating.")
            self.preprocess(kb_model, kb_process_gpus, kb_process_batch_size_per_gpu)
        self.open_file()

    @property
    def train_dataset(self):
        return StaticIterableDataset(
            len(self.file[self.task]["train"]["idx"]), self.generator, ("train",),
        )

    @property
    def validate_dataset(self):
        return StaticIterableDataset(
            self.validate_size, self.generator, ("validation",)
        )

    @property
    def test_dataset(self):
        return StaticIterableDataset(self.test_size, self.generator, ("test",))

    @property
    def validate_size(self):
        if self.task != "mnli":
            return len(self.file[self.task]["validation"]["idx"])
        else:
            return len(self.file["mnli"]["validation_matched"]["idx"]) + len(
                self.file["mnli"]["validation_mismatched"]["idx"]
            )

    @property
    def test_size(self):
        if self.task != "mnli":
            return len(self.file[self.task]["test"]["idx"])
        else:
            return (
                len(self.file["mnli"]["test_matched"]["idx"])
                + len(self.file["mnli"]["test_mismatched"]["idx"])
                + len(self.file["ax"]["test"]["idx"])
            )

    def generator(self, index: int, split: str):
        self.open_file()
        if self.task != "mnli" or split == "train":
            dataset = self.task
            split = split
        else:
            if split == "validation":
                dataset = self.task
                split, index = self.find_part(
                    [
                        len(self.file["mnli"]["validation_matched"]["idx"]),
                        len(self.file["mnli"]["validation_mismatched"]["idx"]),
                    ],
                    ["validation_matched", "validation_mismatched"],
                    index,
                )
            else:
                split, index = self.find_part(
                    [
                        len(self.file["mnli"]["test_matched"]["idx"]),
                        len(self.file["mnli"]["test_mismatched"]["idx"]),
                        len(self.file["ax"]["test"]["idx"]),
                    ],
                    ["test_matched", "test_mismatched", "test"],
                    index,
                )
                dataset = "ax" if split == "test" else "mnli"

        kb_embeds = t.from_numpy(self.file[dataset][split]["kb_embeds"][index])
        input_ids = t.from_numpy(self.file[dataset][split]["input_ids"][index])
        attention_mask = t.from_numpy(
            self.file[dataset][split]["attention_mask"][index]
        )
        token_type_ids = t.from_numpy(
            self.file[dataset][split]["token_type_ids"][index]
        )
        label = self.file[dataset][split]["label"][index]
        idx = self.file[dataset][split]["idx"][index]
        dataset_id = self.file[dataset][split]["dataset_id"][index]
        return {
            "kb_embeds": kb_embeds.unsqueeze(0),
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "token_type_ids": token_type_ids.unsqueeze(0),
            "label": t.tensor(
                [label], dtype=t.long if self.task != "stsb" else t.float32
            ),
            "idx": t.tensor([idx], dtype=t.long),
            "dataset_id": t.tensor([dataset_id], dtype=t.long),
        }

    @staticmethod
    def find_part(part_lengths: List[int], part_names: List[str], index: int):
        for part_index, part_length in enumerate(part_lengths):
            if 0 <= index < part_length:
                return part_names[part_index], index
            index -= part_length
        raise ValueError(
            f"Index {index} is out of range of parts with lengths {part_lengths}"
        )

    def validate(self, batch: BatchEncoding, logits: t.Tensor):
        logits = logits.cpu().numpy()
        # print(logits)
        labels = np.squeeze(logits) if self.is_regression else np.argmax(logits, axis=1)
        ref_labels = batch["label"].cpu().numpy()

        if self.task != "mnli":
            return self.metric.compute(predictions=labels, references=ref_labels)
        else:
            mnli_m_idx = [
                idx
                for idx in range(len(labels))
                if batch["dataset_id"][idx] == self.MNLI_MATCHED
            ]
            mnli_mm_idx = [
                idx
                for idx in range(len(labels))
                if batch["dataset_id"][idx] == self.MNLI_MISMATCHED
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
        if len(labels) != self.test_size:
            raise ValueError(
                f"Label size {len(labels)} does not match test size {self.test_size}"
            )
        # File format is specified by https://gluebenchmark.com/faq FAQ #1
        logging.info("Saving test results.")
        if self.is_regression:
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][0]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels):
                    file.write(f"{index}\t{item:3.3f}\n")
        elif self.task != "mnli":
            label_list = self.datasets[0]["test"].features["label"].names
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][0]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels):
                    file.write(f"{index}\t{label_list[item]}\n")
            with open_file_with_create_directories(
                os.path.join(
                    directory, self.task_to_reports[self.task][0] + ".original"
                ),
                "w",
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels):
                    file.write(f"{index}\t{item}\n")
        else:
            label_list = self.datasets[0]["train"].features["label"].names
            mnli_m_length = len(self.file["mnli"]["test_matched"]["idx"])
            mnli_mm_length = len(self.file["mnli"]["test_mismatched"]["idx"])

            # matched
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][0]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[:mnli_m_length]):
                    file.write(f"{index}\t{label_list[item]}\n")

            with open_file_with_create_directories(
                os.path.join(
                    directory, self.task_to_reports[self.task][0] + ".original"
                ),
                "w",
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[:mnli_m_length]):
                    file.write(f"{index}\t{item}\n")

            # mismatched
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][1]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(
                    labels[mnli_m_length : mnli_m_length + mnli_mm_length]
                ):
                    file.write(f"{index}\t{label_list[item]}\n")

            with open_file_with_create_directories(
                os.path.join(
                    directory, self.task_to_reports[self.task][1] + ".original"
                ),
                "w",
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(
                    labels[mnli_m_length : mnli_m_length + mnli_mm_length]
                ):
                    file.write(f"{index}\t{item}\n")

            # ax
            with open_file_with_create_directories(
                os.path.join(directory, self.task_to_reports[self.task][2]), "w"
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[mnli_m_length + mnli_mm_length :]):
                    file.write(f"{index}\t{label_list[item]}\n")

            with open_file_with_create_directories(
                os.path.join(
                    directory, self.task_to_reports[self.task][2] + ".original"
                ),
                "w",
            ) as file:
                file.write("index\tprediction\n")
                for index, item in enumerate(labels[mnli_m_length + mnli_mm_length :]):
                    file.write(f"{index}\t{item}\n")
        logging.info("Saving finished.")

    def preprocess(self, kb_model, kb_process_gpus, kb_process_batch_size_per_gpu):
        data_path = os.path.join(preprocess_cache_dir, "glue_data", f"{self.task}.hdf5")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with h5py.File(data_path, "w", rdcc_nbytes=1024 ** 3) as file:
            hidden_size = kb_model.hidden_size
            kb_model = t.nn.DataParallel(
                KBProxy(kb_model).to(kb_process_gpus[0]),
                device_ids=kb_process_gpus,
                output_device=kb_process_gpus[0],
            )
            logging.info("Parallel kb models created.")
            # Pre-processing the raw_datasets
            sentence1_key, sentence2_key = self.task_to_keys[self.task]
            for dataset, dataset_name in zip(
                self.datasets, self.task_to_datasets[self.task]
            ):
                group = file.create_group(dataset_name)
                for sub_dataset_name in dataset:
                    logging.info(f"Processing {sub_dataset_name}")

                    limit_num = None
                    if "train" in sub_dataset_name:
                        limit_num = self.max_train_samples
                    elif "validation" in sub_dataset_name:
                        limit_num = self.max_validate_samples
                    elif "test" in sub_dataset_name:
                        limit_num = self.max_test_samples
                    if limit_num is not None:
                        if limit_num <= 0:
                            raise ValueError(
                                f"Select number must be greater than 0, "
                                f"but got {limit_num}"
                            )
                        limit_num = min(len(dataset[sub_dataset_name]), limit_num)
                    else:
                        limit_num = len(dataset[sub_dataset_name])

                    sub_group = group.create_group(sub_dataset_name)
                    kb_embeds_dataset = sub_group.create_dataset(
                        name="kb_embeds",
                        shape=(limit_num, self.max_seq_length, hidden_size),
                        dtype=np.float32,
                        chunks=(16, self.max_seq_length, hidden_size),
                    )
                    input_ids_dataset = sub_group.create_dataset(
                        name="input_ids",
                        shape=(limit_num, self.max_seq_length),
                        dtype=np.int32,
                    )
                    attention_mask_dataset = sub_group.create_dataset(
                        name="attention_mask",
                        shape=(limit_num, self.max_seq_length),
                        dtype=np.float32,
                    )
                    token_type_ids_dataset = sub_group.create_dataset(
                        name="token_type_ids",
                        shape=(limit_num, self.max_seq_length),
                        dtype=np.int32,
                    )
                    label_dataset = sub_group.create_dataset(
                        name="label",
                        shape=(limit_num,),
                        dtype=np.int32 if self.task != "stsb" else np.float32,
                    )
                    idx_dataset = sub_group.create_dataset(
                        name="idx", shape=(limit_num,), dtype=np.int32
                    )
                    dataset_id_dataset = sub_group.create_dataset(
                        name="dataset_id", shape=(limit_num,), dtype=np.int32,
                    )

                    with tqdm(
                        total=limit_num, desc="Processed samples", unit=" samples"
                    ) as progress_bar:
                        for i in range(
                            0,
                            limit_num,
                            kb_process_batch_size_per_gpu * len(kb_process_gpus),
                        ):
                            processed_num = min(
                                kb_process_batch_size_per_gpu * len(kb_process_gpus),
                                limit_num - i,
                            )
                            examples = dataset[sub_dataset_name][i : i + processed_num]
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
                                return_tensors="np",
                            )

                            with t.no_grad():
                                kb_embeds = (
                                    kb_model(
                                        t.tensor(encodings["input_ids"], dtype=t.long,),
                                        context_length=self.kb_context_length,
                                        sequence_length=self.kb_max_seq_length,
                                        process_batch_size=kb_process_batch_size_per_gpu,
                                    )
                                    .to("cpu")
                                    .numpy()
                                )

                            input_ids_dataset[i : i + processed_num] = encodings[
                                "input_ids"
                            ]
                            kb_embeds_dataset[i : i + processed_num] = kb_embeds
                            attention_mask_dataset[i : i + processed_num] = encodings[
                                "attention_mask"
                            ]
                            token_type_ids_dataset[i : i + processed_num] = encodings[
                                "token_type_ids"
                            ]
                            label_dataset[i : i + processed_num] = examples["label"]
                            idx_dataset[i : i + processed_num] = examples["idx"]

                            if self.task == "mnli":
                                if (
                                    dataset_name == "mnli"
                                    and "_matched" in sub_dataset_name
                                ):
                                    dataset_id_dataset[
                                        i : i + processed_num
                                    ] = self.MNLI_MATCHED
                                elif (
                                    dataset_name == "mnli"
                                    and "_mismatched" in sub_dataset_name
                                ):
                                    dataset_id_dataset[
                                        i : i + processed_num
                                    ] = self.MNLI_MISMATCHED
                                if dataset_name == "ax":
                                    dataset_id_dataset[i : i + processed_num] = self.AX
                            else:
                                dataset_id_dataset[i : i + processed_num] = -1
                            progress_bar.update(processed_num)

    def open_file(self):
        data_path = os.path.join(preprocess_cache_dir, "glue_data", f"{self.task}.hdf5")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        self.file = h5py.File(data_path, "r", rdcc_nbytes=4 * 1024 ** 3)
