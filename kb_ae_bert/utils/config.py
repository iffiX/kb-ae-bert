import json
from pydantic import BaseModel, validator
from typing import *


class KBEncoderTrainConfig(BaseModel):
    seed: int = 0
    base_type: str = "bert-base-uncased"
    relation_mode: str = "concatenation"
    mlp_hidden_size: Tuple[int] = ()


class QATrainConfig(BaseModel):
    seed: int = 0
    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: Optional[float]
    batch_size: int = 256
    base_type: str = "bert-base-uncased"
    extend_config: Optional[Dict[str, Any]]
    extend_mode: str = "ratio_mix"
    base_configs: Dict[str, Any] = {}
    kb_encoder_trainable: bool = False
    dataset_name: str = "nq"


class Config(BaseModel):
    pipeline: List[str] = ["kb_encoder", "qa"]
    configs: List[Union[QATrainConfig, KBEncoderTrainConfig]]


def load_config(path: str):
    with open(path, "r") as f:
        return Config(**json.load(f))


def save_config(config: Config, path: str):
    with open(path, "w") as f:
        json.dump(config.dict(), f)
