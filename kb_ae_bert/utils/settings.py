import os
from typing import Union

ROOT = os.path.dirname(os.path.abspath(__file__))
http_proxy = None  # type: Union[str, None]
kaggle_username = ""  # type: str
kaggle_key = ""  # type: str
model_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, os.pardir, "data", "model"))
)  # type: str
dataset_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, os.pardir, "data", "dataset"))
)  # type: str


def reset():
    # init kaggle
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    if http_proxy is not None:
        os.environ["KAGGLE_PROXY"] = http_proxy
