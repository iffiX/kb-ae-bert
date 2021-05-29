import torch as t
from torch.utils.data import Dataset
from transformers import BatchEncoding


class TorchDataset(Dataset):
    def __init__(self, encodings: BatchEncoding):
        self.encodings = encodings

    def __getitem__(self, idx: int):
        return {key: t.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
