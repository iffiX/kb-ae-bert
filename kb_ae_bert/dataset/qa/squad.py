import os
import logging
import torch as t
from .base import QADataset, StaticMapDataset
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase, BatchEncoding
from datasets import load_dataset, load_metric, DownloadConfig
from kb_ae_bert.utils.settings import dataset_cache_dir, metrics_cache_dir, proxies


class SQuADDataset(QADataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str = "squad",
        local_root_path: str = None,
    ):
        local_root_path = local_root_path or str(
            os.path.join(dataset_cache_dir, "huggingface")
        )
        self.dataset = load_dataset(
            path=dataset_path,
            cache_dir=local_root_path,
            download_config=DownloadConfig(proxies=proxies),
        )

        # squad v2 works for squad and squad v2 and any custom squad datasets
        self.metric = load_metric(
            "squad_v2",
            cache_dir=metrics_cache_dir,
            download_config=DownloadConfig(proxies=proxies),
        )
        self.tokenizer = tokenizer
        self._train = None
        self._validate = None

    @property
    def train_dataset(self):
        # lazy preprocess
        if self._train is None:
            self._train = self.preprocess(split="train")
        return StaticMapDataset(self._train)

    @property
    def validate_dataset(self):
        if self._validate is None:
            self._validate = self.preprocess(split="validate")
        return StaticMapDataset(self._validate)

    def validate(self, batch, start_logits, end_logits):
        if self._validate is None:
            self._validate = self.preprocess(split="validate")
        predictions = []
        references = []
        for index, input_ids, start_l, end_l in zip(
            batch["sample-index"], batch["input_ids"], start_logits, end_logits
        ):
            sample = self._validate[index]
            # get the most likely beginning of answer with the argmax of the score
            answer_start = t.argmax(start_l)
            answer_end = t.argmax(end_l) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            )
            for answer_ref in sample["answers"]["text"]:
                predictions.append(
                    {
                        "prediction_text": answer,
                        "id": sample["id"],
                        "no_answer_probability": 0.0,
                    }
                )
                references.append(
                    {
                        "prediction_text": answer_ref,
                        "id": sample["id"],
                        "no_answer_probability": 0.0,
                    }
                )

        return self.metric.compute(predictions=predictions, references=references)

    def preprocess(self, split="train"):
        logging.info(f"SQuADDataset begin pre-processing split {split}")
        # flatten answers in the dataset
        contexts = []
        questions = []
        answers = []
        indexes = []
        for idx, item in enumerate(self.dataset[split]):
            """
            https://huggingface.co/datasets/squad
            {
                "answers": {
                    "answer_start": [1],
                    "text": ["This is a test text"]
                },
                "context": "This is a test context.",
                "id": "1",
                "question": "Is this a test?",
                "title": "train test"
            }
            """
            num_answers = len(item["answers"]["text"])
            for answer_idx in range(num_answers):
                indexes.append(idx)
                contexts.append(item["context"])
                questions.append(item["question"])
                answers.append({
                    "answer_start":item["answers"]["answer_start"][answer_idx],
                    "text":item["answers"]["text"][answer_idx]
                    })

        # add end idx to answers
        self.add_end_idx(answers, contexts)

        # compute encoding [cls] context [sep] answer [sep]
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)

        # update token start / end positions
        self.add_token_positions(encodings, self.tokenizer, answers)

        # update index
        encodings.update({"sample-index": indexes})

        logging.info(f"SQuADDataset finished pre-processing split {split}")
        return encodings

    @staticmethod
    def add_end_idx(answers: List[Dict[str, Any]], contexts: List[str]):
        # print(answers)
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]
            start_idx = answer["answer_start"]
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            elif context[start_idx - 1 : end_idx - 1] == gold_text:
                # When the gold label is off by one character
                answer["answer_start"] = start_idx - 1
                answer["answer_end"] = end_idx - 1
            elif context[start_idx - 2 : end_idx - 2] == gold_text:
                # When the gold label is off by two characters
                answer["answer_start"] = start_idx - 2
                answer["answer_end"] = end_idx - 2

    @staticmethod
    def add_token_positions(
        encodings: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        answers: List[Dict[str, Any]],
    ):
        start_positions = []
        end_positions = []
        for i, answer in enumerate(answers):
            start_positions.append(encodings.char_to_token(i, answer["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answer["answer_end"] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

        encodings.update(
            {"start_positions": start_positions, "end_positions": end_positions}
        )
