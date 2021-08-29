import os
from typing import Union

# settings.py is for global configs that do not differentiate
# between different trainings.

# ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
# ROOT = "/nfs/vdisk/workspace/kb-ae-bert"

ROOT = "/root/data/workspace/kb-ae-bert"

# in requests format
# proxies = {
#     "http": "http://localhost:1090",
#     "https": "http://localhost:1090",
# }  # type: Union[dict, None]

proxies = None

# kaggle_http_proxy = "http://localhost:1090"
kaggle_http_proxy = None
kaggle_username = ""  # type: str
kaggle_key = ""  # type: str
model_cache_dir = str(os.path.abspath(os.path.join(ROOT, "data", "model")))  # type: str
dataset_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, "data", "dataset"))
)  # type: str
metrics_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, "data", "metrics"))
)  # type: str
preprocess_cache_dir = str(
    os.path.abspath(os.path.join(ROOT, "data", "preprocess"))
)  # type: str
mongo_config = {
    "is_docker": False,
    "mongo_local_path": str(os.path.abspath(os.path.join(ROOT, "data", "mongo"))),
}
preprocess_worker_num = 16


def reset():
    # init kaggle
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    if kaggle_http_proxy is not None:
        os.environ["KAGGLE_PROXY"] = kaggle_http_proxy
