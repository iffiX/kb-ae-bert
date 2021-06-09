import json
from pydantic import BaseModel
from typing import *


class KDWDConfig(BaseModel):
    graph_depth: int = 1
    permute_seed: int = 0
    train_entity_encode_ratio: float = 0.9
    train_relation_encode_ratio: float = 0.9
    local_root_path: str = None
    mongo_docker_name: str = None
    mongo_docker_host: str = "localhost"
    mongo_docker_api_host: str = None
    force_reload: bool = False


class KBEncoderTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    epochs: int = 100
    train_steps: int = 10000
    validate_steps: int = 100
    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: Optional[float]
    relation_size: int = 200
    context_length: int = 200
    batch_size: int = 256
    # The process of generating samples for KB is complex
    # and needs multiprocessing.
    # This is the number of workers used in Dataloader.
    load_worker_num: int = 4

    base_type: str = "bert-base-uncased"
    base_configs: Dict[str, Any] = {}
    relation_mode: str = "concatenation"
    mlp_hidden_size: List[int] = []

    # "entity" or "relation"
    task: str = "entity"
    dataset: str = "KDWD"
    dataset_config: KDWDConfig = KDWDConfig()


class QATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    epochs: int = 100
    train_steps: int = 10000
    validate_steps: int = 100
    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: Optional[float]
    context_length: int = 200
    batch_size: int = 256

    base_type: str = "bert-base-uncased"
    extend_config: Optional[Dict[str, Any]]
    extend_mode: str = "ratio_mix"
    base_configs: Dict[str, Any] = {}

    kb_encoder_path: str = ""
    kb_encoder_trainable: bool = False
    # "squad", "squad_v2", "nq", etc.
    train_dataset_path: Optional[str] = "squad"
    validate_dataset_path: Optional[str] = "squad"


class Config(BaseModel):
    # Cuda ids of GPUs
    gpus: Optional[List[int]] = [0]

    # Maximum train steps allowed before stopping
    # when monitored metric is not decreasing
    early_stopping_patience: int = 100

    # Path to the working directory
    # sub-stages will be created as 0, 1, ... subdirectories
    working_directory: str = "./train"

    # example: ["kb_encoder", "qa"]
    # config in configs must match items in pipeline
    pipeline: List[str] = []
    configs: List[Union[QATrainConfig, KBEncoderTrainConfig]] = []


def load_config(path: str):
    with open(path, "r") as f:
        config_dict = json.load(f)
        config = Config(
            gpus=config_dict["gpus"],
            early_stopping_patience=config_dict["early_stopping_patience"],
            working_directory=config_dict["working_directory"],
        )
        for p, c in zip(config_dict["pipeline"], config_dict["configs"]):
            config.pipeline.append(p)
            if p == "kb_encoder":
                config.configs.append(KBEncoderTrainConfig(**c))
            elif p == "qa":
                config.configs.append(QATrainConfig(**c))
            else:
                raise ValueError(f"Unknown stage {p}.")
        return config


def save_config(config: Config, path: str):
    with open(path, "w") as f:
        json.dump(config.dict(), f, indent=4, sort_keys=True)
