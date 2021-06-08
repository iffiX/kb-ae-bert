import os
from typing import Union

# settings.py is for global configs that do not differentiate
# between different trainings.

ROOT = os.path.dirname(os.path.abspath(__file__))
# in requests format
proxies = {
    "http": "socks5://localhost:1082",
    "https": "socks5://localhost:1082",
}  # type: Union[dict, None]
kaggle_http_proxy = "http://localhost:3128"
kaggle_username = ""  # type: str
kaggle_key = ""  # type: str
model_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, os.pardir, "data", "model"))
)  # type: str
dataset_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, os.pardir, "data", "dataset"))
)  # type: str
metrics_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, os.pardir, "data", "metrics"))
)  # type: str
preprocess_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, os.pardir, "data", "preprocess"))
)  # type: str
mongo_docker_name = "mongodb2"


def reset():
    # init kaggle
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    if kaggle_http_proxy is not None:
        os.environ["KAGGLE_PROXY"] = kaggle_http_proxy
