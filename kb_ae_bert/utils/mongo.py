import os
import subprocess
import logging
import pymongo as mon
from typing import List, Union

try:
    from urllib.parse import quote_plus
except ImportError:
    from urllib import quote_plus

from docker.types import Mount
from .docker import (
    safe_exec_on_docker,
    create_or_reuse_docker,
    allocate_port,
    Container,
)


def load_dataset_files(database: str, dataset_files_path: str, files: List[str]):
    for file in files:
        file_name, ext = os.path.splitext(file)
        logging.info(f"Importing collection {file_name}")
        if "csv" in ext:
            subprocess.run(
                [
                    "mongoimport",
                    "--db",
                    database,
                    "--collection",
                    file_name,
                    "--drop",
                    "--file",
                    str(os.path.join(dataset_files_path, file)),
                    "--type",
                    "csv",
                    "--headerline",
                ]
            )
        elif "json" in ext:
            subprocess.run(
                [
                    "mongoimport",
                    "--db",
                    database,
                    "--collection",
                    file_name,
                    "--drop",
                    "--file",
                    str(os.path.join(dataset_files_path, file)),
                    "--type",
                    "json",
                ]
            )
        else:
            raise ValueError(f"Unsupported extension {ext}")


def load_dataset_files_to_docker(
    container: Container, database: str, dataset_files_path: str, files: List[str]
):
    for file in files:
        file_name, ext = os.path.splitext(file)
        logging.info(f"Importing collection {file_name}")
        if "csv" in ext:
            safe_exec_on_docker(
                container,
                f"mongoimport "
                f"--db {database} "
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
                f"--db {database} "
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
    client = mon.MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client.get_database(db_name)


class MongoDBHandler:
    def __init__(
        self,
        is_docker: bool = True,
        mongo_local_path: str = "/var/lib/mongodb",
        mongo_docker_name: str = "mongodb",
        mongo_docker_host: str = "localhost",
        mongo_docker_api_host: str = None,
        mongo_docker_dataset_path: str = None,
    ):
        """
        Args:
            is_docker: Whether use docker to create a new instance of the MongoDB or
                use an existing local one.
            mongo_docker_name: Name of the created / reused mongo docker.
            mongo_docker_host: Host address of the mongo docker.
            mongo_docker_api_host: Host address of the docker API, could be different
                from mongo_docker_host if you are using a cluster, etc.
            mongo_docker_dataset_path: Local dataset directory which will be mount to
                "/mnt/dataset".
        """
        self.is_docker = is_docker

        if is_docker:
            logging.info("Initializing MongoDB using docker.")
            if mongo_docker_api_host is not None:
                os.environ["DOCKER_HOST"] = mongo_docker_api_host

            if mongo_docker_dataset_path is not None:
                mounts = [
                    Mount(
                        target="/mnt/dataset",
                        source=mongo_docker_dataset_path,
                        type="bind",
                        read_only=True,
                    )
                ]
            else:
                mounts = []

            self.db_docker, _ = create_or_reuse_docker(
                image="mongo:latest",
                startup_args={"ports": {"27017": allocate_port()}, "mounts": mounts,},
                reuse_name=mongo_docker_name,
            )
            self.db_host = mongo_docker_host
            self.db_port = int(
                self.db_docker.attrs["HostConfig"]["PortBindings"]["27017/tcp"][0][
                    "HostPort"
                ]
            )
        else:
            logging.info("Initializing MongoDB using local binary.")
            os.makedirs(mongo_local_path, exist_ok=True)
            self.mongo_local_path = mongo_local_path
            self.db_host = "localhost"
            self.db_port = 27017
            try:
                client = mon.MongoClient(
                    f"mongodb://localhost:27017", serverSelectionTimeoutMS=500
                )
                client.server_info()
                self.db_process = None
            except:
                self.db_process = subprocess.Popen(
                    [
                        "mongod",
                        "--dbpath",
                        self.mongo_local_path,
                        "--bind_ip",
                        "127.0.0.1",
                        "--port",
                        "27017",
                        "--wiredTigerCacheSizeGB",
                        "20",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )

    def get_database(self, database):
        return connect_to_database(self.db_host, self.db_port, database)

    def load_dataset_files(
        self, database: str, dataset_files_path: str, files: List[str]
    ):
        if self.is_docker:
            load_dataset_files_to_docker(
                self.db_docker, database, dataset_files_path, files
            )
        else:
            load_dataset_files(database, dataset_files_path, files)

    def stop(self):
        if self.is_docker:
            self.db_docker.stop()
        else:
            if self.db_process is not None:
                self.db_process.terminate()
                self.db_process.wait()
                self.db_process = None

    def start(self):
        if self.is_docker:
            self.db_docker.start()
        else:
            if self.db_process is None:
                self.db_process = subprocess.Popen(
                    [
                        "mongod",
                        "--dbpath",
                        self.mongo_local_path,
                        "--bind_ip",
                        "127.0.0.1",
                        "--port",
                        "27017",
                        "--wiredTigerCacheSizeGB",
                        "20",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
