import os
import torch as t
from urllib.parse import urlparse
from transformers import BatchEncoding
from datasets import load_dataset
from datasets.utils import DownloadConfig
from kb_ae_bert.utils.settings import dataset_cache_dir, proxies
from ..base import TorchDataset


class QADataset:
    def __init__(
        self,
        dataset_path: str,
        dataset_config: str = None,
        local_root_path: str = None,
    ):
        """
        Args:
            dataset_path: Name of the target dataset, or a local path.
            dataset_config: Name of the used dataset config.
            local_root_path: Local root path used for caching.
        """
        local_root_path = local_root_path or str(
            os.path.join(dataset_cache_dir, "huggingface")
        )
        self.dataset = load_dataset(
            path=dataset_path,
            name=dataset_config,
            cache_dir=local_root_path,
            download_config=DownloadConfig(proxies=proxies),
        )

    def validate(
        self, batch: BatchEncoding, start_logits: t.Tensor, end_logits: t.Tensor
    ):
        """
        Args:
            batch: Sampled batch.
            start_logits: FloatTensor of shape (batch_size, sequence_length),
                Span-start scores (before SoftMax).
            end_logits: FloatTensor of shape (batch_size, sequence_length),
                Span-end scores (before SoftMax).

        Returns:
            A dictionary of various metrics that will be logged by the validating_step.
        """
        pass

    @property
    def train_dataset(self) -> TorchDataset:
        raise NotImplementedError

    @property
    def validate_dataset(self) -> TorchDataset:
        raise NotImplementedError
