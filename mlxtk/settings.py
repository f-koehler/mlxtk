import os
from typing import Any, Dict

import yaml

from .util import memoize


@memoize
def load_settings(start_directory: str = os.getcwd()) -> Dict[Any, Any]:
    start_directory = os.path.abspath(os.path.realpath(start_directory))
    while True:
        path = os.path.join(start_directory, ".mlxtkrc")
        if os.path.exists(path):
            with open(path, "r") as fp:
                return yaml.load(fp, Loader=yaml.CLoader)

        new_directory = os.path.split(start_directory)[0]
        if new_directory == start_directory:
            return {}

        start_directory = new_directory
