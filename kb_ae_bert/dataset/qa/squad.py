import logging
from .base import QADataset, TorchDataset
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase, BatchEncoding


class SQuADDataset(QADataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, local_root_path: str = None,
    ):
        super().__init__("squad", local_root_path=local_root_path)
        self.tokenizer = tokenizer
        self._train = None
        self._validate = None

    @property
    def train_dataset(self) -> TorchDataset:
        # lazy preprocess
        if self._train is None:
            self._train = self.preprocess(split="train")
        return TorchDataset(self._train)

    @property
    def validate_dataset(self) -> TorchDataset:
        if self._validate is None:
            self._validate = self.preprocess(split="validate")
        return TorchDataset(self._validate)

    def preprocess(self, split="train"):
        logging.info(f"SQuADDataset begin pre-processing split {split}")
        # flatten answers in the dataset
        contexts = []
        questions = []
        answers = []
        for item in self.dataset[split]:
            for answer in item["answers"]:
                contexts.append(item["context"])
                questions.append(item["question"])
                answers.append(answer)

        # add end idx to answers
        self.add_end_idx(answers, contexts)

        # compute encoding [cls] context [sep] answer [sep]
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)

        # update token start / end positions
        self.add_token_positions(encodings, self.tokenizer, answers)
        logging.info(f"SQuADDataset finished pre-processing split {split}")
        return encodings

    @staticmethod
    def add_end_idx(answers: List[Dict[str, Any]], contexts: List[str]):
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
