import os
import errno


def open_file_with_create_directories(path, mode):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)
