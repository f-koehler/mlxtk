from pathlib import Path
from typing import Any, Dict

import yaml

from .util import memoize


@memoize
def load_settings(start_directory: Path = Path.cwd()) -> Dict[Any, Any]:
    start_directory = start_directory.resolve()
    while True:
        path = start_directory / ".mlxtkrc"
        if path.exists():
            with open(path, "r") as fp:
                settings = yaml.load(fp, Loader=yaml.CLoader)

            if "paths" in settings:
                for path in settings["paths"]:
                    settings["paths"][path] = (
                        Path(start_directory) /
                        Path(settings["paths"][path])).resolve()

            return settings

        new_directory = start_directory.parent
        if new_directory == start_directory:
            return {}

        start_directory = new_directory


@memoize
def load_path(name: str, start_directory: Path = Path.cwd()) -> Path:
    settings = load_settings(start_directory)
    if not "paths" in settings:
        raise ValueError("no paths stored in settings")
    if not name in settings["paths"]:
        raise ValueError("path \"{}\" not stored in settings".format(name))
    return settings["paths"][name]
