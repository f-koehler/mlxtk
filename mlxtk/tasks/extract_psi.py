import os.path
from typing import Any, Callable, Dict, Iterable, List

from ..inout.psi import read_psi_hdf5, write_psi_ascii
from ..log import get_logger

LOGGER = get_logger(__name__)


def extract_psi(name: str) -> List[Callable]:
    path_compressed = name + ".hdf5"

    def task_extract() -> Dict[str, Any]:
        def action_extract(targets: Iterable[str]):
            if os.path.exists(name):
                LOGGER.info("ascii psi file already present, skipping")
                return

            LOGGER.info("extract psi file")
            write_psi_ascii(name, read_psi_hdf5(path_compressed))

        return {
            "name": "extract_psi:{}:extract".format(name),
            "actions": [action_extract],
            "targets": [name],
            "file_dep": [path_compressed],
        }

    return [task_extract]


class ExtractedPsi(object):
    def __init__(self, simulation, psi):
        self.simulation = simulation
        self.psi = psi

    def __enter__(self):
        self.simulation += extract_psi(self.psi)

    def __exit__(self, type_, value, tb):
        def action_remove(targets):
            if os.path.exists(self.psi):
                LOGGER.info("remove ascci psi file")
                os.remove(self.psi)

        def task_remove():
            return {
                "name": "extract_psi:{}:remove".format(self.psi),
                "actions": [action_remove],
            }

        self.simulation += [task_remove]
