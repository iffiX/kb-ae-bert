import os
import logging
import pymongo as mon
import pymongo.errors as mon_err
from typing import List, Mapping, Union, Any

try:
    from urllib.parse import quote_plus
except ImportError:
    from urllib import quote_plus

from typing import List
from .docker import safe_exec_on_docker, Container


def load_dataset_files(
    container: Container, db_name: str, dataset_files_path: str, files: List[str]
):
    for file in files:
        _, ext = os.path.splitext(file)
        file_name = os.path.basename(file)
        if "csv" in ext:
            safe_exec_on_docker(
                container,
                f"mongoimport "
                f"--db {db_name} "
                f"--collection {file_name} "
                f"--drop "
                f"--file {str(os.path.join(dataset_files_path, file))}"
                f"--type csv"
                f"--headerline",
            )
        elif "json" in ext:
            safe_exec_on_docker(
                container,
                f"mongoimport "
                f"--db {db_name} "
                f"--collection {file_name} "
                f"--drop "
                f"--file {str(os.path.join(dataset_files_path, file))}"
                f"--type json",
            )
        else:
            raise ValueError(f"Unsupported extension {ext}")


def connect_to_database(
    host: str,
    port: Union[int, str],
    db_name: str,
    username: str = None,
    password: str = None,
):
    if username is not None and password is not None:
        uri = f"mongodb://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}"
    else:
        uri = f"mongodb://{host}:{port}"
    client = mon.MongoClient(uri, serverSelectionTimeoutMS=3000)
    return client.get_database(db_name)
