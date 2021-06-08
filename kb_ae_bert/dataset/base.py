import torch as t
from typing import Callable, Dict
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding


class StaticMapDataset(Dataset):
    def __init__(self, encodings: BatchEncoding):
        self.encodings = encodings

    def __getitem__(self, idx: int):
        return {key: t.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class DynamicIterableDataset(IterableDataset):
    def __init__(self, generator: Callable[[], Dict[str, t.Tensor]]):
        self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return self.generator()


class EmptyDataset(Dataset):
    def __getitem__(self, idx: int):
        return None

    def __len__(self):
        return 0
