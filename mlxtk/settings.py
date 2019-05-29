from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .util import memoize


@memoize
def find_settings(start_directory: Path = Path.cwd()) -> Optional[Path]:
    start_directory = start_directory.resolve()
    while True:
        path = start_directory / "mlxtkrc"
        if path.exists():
            return path

        new_directory = start_directory.parent
        if new_directory == start_directory:
            return None

        start_directory = new_directory


@memoize
def load_settings(start_directory: Path = Path.cwd()) -> Dict[Any, Any]:
    settings_path = find_settings()
    if settings_path is None:
        return {}

    with open(settings_path, "r") as fp:
        settings = yaml.load(fp, Loader=yaml.CLoader)

    if "paths" in settings:
        for path in settings["paths"]:
            settings["paths"][path] = (
                Path(start_directory) /
                Path(settings["paths"][path])).resolve()

    if "env" in settings:
        for entry in settings["env"]:
            if isinstance(settings["env"], list):
                settings["env"][entry] = [
                    (settings_path.parent / Path(path)).resolve()
                    for path in settings["env"][entry]
                ]
            else:
                if settings["env"][entry].startswith("+"):
                    settings["env"][entry] = [
                        (settings_path.parent /
                         Path(settings["env"][entry][1:])).resolve()
                    ]

    return settings


def load_path(name: str, start_directory: Path = Path.cwd()) -> Path:
    settings = load_settings(start_directory)
    if not "paths" in settings:
        raise ValueError("no paths stored in settings")
    if not name in settings["paths"]:
        raise ValueError("path \"{}\" not stored in settings".format(name))
    return settings["paths"][name]


def load_full_env(start_directory: Path = Path.cwd()) -> Dict[str, Path]:
    settings = load_settings(start_directory)
    return settings.get("env", {})
