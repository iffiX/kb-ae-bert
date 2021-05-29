import os
import kaggle
import zipfile


def download_dataset(dataset: str, local_root_path: str):
    """
    Download (if not cached in `local_root_path`) and extract kaggle dataset.

    Args:
        dataset: the string identified of the dataset
            should be in format [owner]/[dataset-name].
        local_root_path: Local save path for all datasets, each dataset will be
            saved in its respective directory with the same [dataset-name].
    """
    owner_name, dataset_name = dataset.split("/")
    kaggle.api.authenticate()
    local_path = os.path.join(local_root_path, dataset_name)
    kaggle.api.dataset_download_files(
        dataset=dataset, path=local_path,
    )
    if "/" not in dataset:
        raise ValueError(f"Invalid dataset name: {dataset}")
    dataset_file = os.path.join(local_path, dataset_name + ".zip")

    with zipfile.ZipFile(dataset_file) as zip_file:
        for member in zip_file.namelist():
            if not (
                os.path.exists(f"{local_path}/{member}")
                or os.path.isfile(f"{local_path}/{member}")
            ):
                zip_file.extract(member, local_path)
