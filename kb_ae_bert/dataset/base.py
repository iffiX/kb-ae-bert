import torch as t
from typing import Callable, Dict, List
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


def collate_function_dict_to_batch_encoding(samples: List[Dict[str, t.Tensor]]):
    assert isinstance(samples, list)
    assert len(samples) > 0
    keys = set(samples[0].keys())
    for other_sample in samples[1:]:
        other_keys = set(other_sample.keys())
        if not other_keys == keys:
            raise ValueError(f"Keys are different: {keys} and {other_keys}")

    result = {}
    for k in keys:
        result[k] = t.cat([s[k] for s in samples], dim=0)
    return BatchEncoding(data=result)
