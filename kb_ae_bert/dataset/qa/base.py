import os
from urllib.parse import urlparse
from datasets import load_dataset
from datasets.utils import DownloadConfig
from kb_ae_bert.utils.settings import dataset_cache_dir, http_proxy
from ..base import TorchDataset


class QADataset:
    def __init__(
        self,
        dataset_name: str,
        dataset_config: str = None,
        local_root_path: str = None,
    ):
        """
        Args:
            dataset_name: Name of the target dataset.
            dataset_config: Name of the used dataset config.
            local_root_path: Local root path used for caching.
        """
        local_root_path = local_root_path or str(
            os.path.join(dataset_cache_dir, "huggingface")
        )
        self.dataset = load_dataset(
            path=dataset_name,
            name=dataset_config,
            cache_dir=local_root_path,
            download_config=DownloadConfig(proxies={"http": urlparse(http_proxy).path}),
        )

    @property
    def train_dataset(self) -> TorchDataset:
        raise NotImplementedError

    @property
    def validate_dataset(self) -> TorchDataset:
        raise NotImplementedError
