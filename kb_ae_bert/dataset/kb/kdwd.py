import os
from docker.types import Mount
from kb_ae_bert.utils.settings import (
    dataset_cache_dir,
    mongo_docker_name as default_mdn,
)
from kb_ae_bert.utils.kaggle import download_dataset
from kb_ae_bert.utils.docker import create_or_reuse_docker, allocate_port
from kb_ae_bert.utils.mongo import load_dataset_files, connect_to_database


class KDWDDataset:
    def __init__(
        self,
        local_root_path: str = None,
        mongo_docker_name: str = None,
        force_reload: bool = False,
    ):
        local_root_path = local_root_path or str(
            os.path.join(dataset_cache_dir, "kaggle")
        )
        mongo_docker_name = mongo_docker_name or default_mdn
        self.db_port = allocate_port()
        self.db_docker, is_reused = create_or_reuse_docker(
            image="mongo:latest",
            startup_args={
                "ports": {"27017": self.db_port},
                "mounts": [
                    Mount(
                        target="/mnt/dataset",
                        source=local_root_path,
                        type="bind",
                        read_only=True,
                    )
                ],
            },
            reuse_name=mongo_docker_name,
        )
        download_dataset(
            "kenshoresearch/kensho-derived-wikimedia-data", local_root_path
        )
        if not is_reused or force_reload:
            # load dataset into the database
            load_dataset_files(
                self.db_docker,
                "kdwd",
                str(os.path.join(local_root_path, "kensho-derived-wikimedia-data")),
                [
                    "item.csv",
                    "item_aliases.csv",
                    "link_annotated_text.jsonl",
                    "page.csv",
                    "property.csv",
                    "property_aliases.csv",
                    "statements.csv",
                ],
            )
        self.db = connect_to_database("localhost", self.db_port, "kdwd")
