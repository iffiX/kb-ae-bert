import os
from docker.types import Mount
from ..utils.kaggle import download_dataset
from ..utils.docker import create_or_reuse_docker, allocate_port
from ..utils.mongo import load_dataset_files, connect_to_database


class KDWDDataset:
    def __init__(self, local_root_path: str, mongo_docker_name: str = None):
        self.db_port = allocate_port()
        self.db_docker = create_or_reuse_docker(
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
        if mongo_docker_name is None:
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
