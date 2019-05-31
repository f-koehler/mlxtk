from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .util import memoize


def create_empty_settings() -> Dict[Any, Any]:
    return {"env": {}, "paths": {}, "units": {}}


@memoize
def find_settings(start_directory: Path = Path.cwd()) -> Optional[List[Path]]:
    start_directory = start_directory.resolve()
    setting_files = []
    while len(start_directory.parts) > 1:
        path = start_directory / "mlxtkrc"
        if path.exists():
            setting_files.append(path)
        start_directory = start_directory.parent
    path = start_directory / "mlxtkrc"
    if path.exists():
        setting_files.append(path)
    return setting_files


def load_settings_file(settings_path: Path) -> Dict[Any, Any]:
    directory = settings_path.parent
    with open(settings_path, "r") as fp:
        settings = yaml.load(fp, Loader=yaml.CLoader)

    if "paths" in settings:
        for path in settings["paths"]:
            settings["paths"][path] = (
                Path(directory) / Path(settings["paths"][path])).resolve()
    else:
        settings["paths"] = {}

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
    else:
        settings["env"] = {}

    settings["units"] = settings.get("units", {})

    return settings


def merge_settings(first: Dict[Any, Any],
                   second: Dict[Any, Any]) -> Dict[Any, Any]:
    new = {key: second[key] for key in second}

    for entry in first["paths"]:
        new["paths"][entry] = first["paths"][entry]

    for entry in first["env"]:
        if isinstance(first["env"][entry], list):
            new["env"][entry] = first["env"][entry] + new["env"].get(entry, [])
        else:
            new["env"][entry] = first["env"][entry]

    for entry in first["units"]:
        new["units"][entry] = first["units"][entry]

    return new


@memoize
def load_settings(start_directory: Path = Path.cwd()) -> Dict[Any, Any]:
    files = find_settings(start_directory)
    settings = create_empty_settings()

    for file_ in files:
        settings = merge_settings(settings, load_settings_file(file_))

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
