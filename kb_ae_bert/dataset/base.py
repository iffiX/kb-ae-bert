import torch as t
from typing import Callable, Dict, List, Union, Any
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding

import traceback


class StaticMapDataset(Dataset):
    def __init__(self, encodings: BatchEncoding):
        self.encodings = encodings

    def __getitem__(self, idx: int):
        return {
            key: t.tensor(val[idx]).unsqueeze(0) for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings.input_ids)


class StaticIterableDataset(Dataset):
    def __init__(
        self,
        length: int,
        generator: Callable[..., Dict[str, t.Tensor]],
        generator_args: tuple = (),
    ):
        self.length = length
        self.generator = generator
        self.generator_args = generator_args

    def __getitem__(self, idx: int):
        return self.generator(idx, *self.generator_args)

    def __len__(self):
        return self.length


class DynamicIterableDataset(IterableDataset):
    def __init__(
        self, generator: Callable[..., Dict[str, t.Tensor]], generator_args: tuple = (),
    ):
        self.generator = generator
        self.generator_args = generator_args

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.generator(*self.generator_args)
        except StopIteration:
            traceback.print_exc()
            raise ValueError(
                "The generator thrown a StopIteration exception, shouldn't happen here."
            )
        return result


class EmptyDataset(Dataset):
    def __getitem__(self, idx: int):
        return None

    def __len__(self):
        return 0


class MovableList(list):
    def to(self, *args, **kwargs):
        return self


def collate_function_dict_to_batch_encoding(
    samples: List[Union[BatchEncoding, Dict[str, Any]]]
):
    assert isinstance(samples, list)
    assert len(samples) > 0
    keys = set(samples[0].keys())
    for other_sample in samples[1:]:
        other_keys = set(other_sample.keys())
        if not other_keys == keys:
            raise ValueError(f"Keys are different: {keys} and {other_keys}")

    result = {}
    for k in keys:
        data_list = MovableList()
        data_list.extend([s[k] for s in samples])
        if t.is_tensor(data_list[0]):
            result[k] = t.cat(data_list, dim=0)
        else:
            if isinstance(data_list[0], int):
                result[k] = t.tensor(data_list, dtype=t.int64)
            elif isinstance(data_list[0], float):
                result[k] = t.tensor(data_list, dtype=t.float32)
            elif isinstance(data_list[0], list):
                if isinstance(data_list[0][0], int):
                    result[k] = t.tensor(data_list, dtype=t.int64)
                elif isinstance(data_list[0][0], float):
                    result[k] = t.tensor(data_list, dtype=t.float32)
                else:
                    result[k] = data_list
            else:
                result[k] = data_list
    return BatchEncoding(data=result)
