import json
from pydantic import BaseModel
from typing import *


class KDWDConfig(BaseModel):
    graph_depth: int = 1
    permute_seed: int = 0
    train_entity_encode_ratio: float = 0.9
    train_relation_encode_ratio: float = 0.9
    force_reload: bool = False
    generate_data: bool = True
    entity_number: int = 1000000
    relation_mask_mode: str = "random"
    relation_mask_random_probability: float = 0.15


class KBEncoderTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    epochs: int = 100
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    relation_size: int = 200
    context_length: int = 32
    max_seq_length: int = 64

    # The process of generating samples for KB is complex
    # and needs multiprocessing.
    # This is the number of workers used in Dataloader.
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

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
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    context_length: int = 200

    base_type: str = "bert-base-uncased"
    extend_config: Optional[Dict[str, Any]]
    extend_mode: str = "ratio_mix"
    base_configs: Dict[str, Any] = {}

    kb_encoder_path: str = ""
    kb_encoder_trainable: bool = False
    kb_encoder_with_gradient_num: int = 1
    # "squad", "squad_v2", "nq", etc.
    train_dataset_path: Optional[str] = "squad"
    validate_dataset_path: Optional[str] = "squad"


class GLUETrainConfig(BaseModel):
    task: str = "cola"
    load: bool = False
    seed: int = 0
    epochs: int = 3
    batch_size: int = 2
    accumulate_grad_batches: int = 32
    max_train_samples: Optional[int] = None
    max_validate_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    optimizer_class: str = "Adam"
    learning_rate: float = 2e-5
    l2_regularization: float = 0

    base_type: str = "bert-base-uncased"
    max_seq_length: int = 128
    extend_config: Optional[Dict[str, Any]]
    extend_mode: str = "ratio_mix"
    base_configs: Dict[str, Any] = {}

    kb_encoder_path: str = ""
    kb_encoder_context_length: int = 32
    kb_encoder_max_seq_length: int = 64
    kb_process_gpus: List[int] = [0]
    kb_process_batch_size_per_gpu: int = 32


class Config(BaseModel):
    # Cuda ids of GPUs
    gpus: Optional[List[int]] = [0]

    # Maximum validation epochs allowed before stopping
    # when monitored metric is not decreasing
    early_stopping_patience: int = 100

    # Path to the working directory
    # sub-stages will be created as 0, 1, ... subdirectories
    working_directory: str = "./train"

    # example: ["kb_encoder", "qa"]
    # config in configs must match items in pipeline
    pipeline: List[str] = []
    configs: List[Union[QATrainConfig, KBEncoderTrainConfig, GLUETrainConfig]] = []


def load_config(path: str) -> Config:
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
            elif p == "glue":
                config.configs.append(GLUETrainConfig(**c))
            else:
                raise ValueError(f"Unknown stage {p}.")
        return config


def save_config(config: Config, path: str):
    with open(path, "w") as f:
        json.dump(config.dict(), f, indent=4, sort_keys=True)
