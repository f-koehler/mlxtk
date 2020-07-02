import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy

from mlxtk.cwd import WorkingDir
from mlxtk.doit_compat import DoitAction
from mlxtk.inout.psi import read_psi_frame_ascii, write_psi_ascii
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks.task import Task
from mlxtk.util import make_path
from mlxtk.wave_function_db import MissingWfnError, load_db

LOGGER = get_logger(__name__)


class RequestWaveFunction(Task):
    def __init__(
        self,
        name: str,
        parameters: Parameters,
        db_path: Union[str, Path],
        variable_name: str = "db",
        compute: bool = True,
    ):

        db_path = make_path(db_path)
        if not db_path.is_absolute():
            self.db_path = Path(sys.argv[0]).resolve().parent / db_path
        else:
            self.db_path = db_path

        self.name = name
        self.parameters = parameters
        self.variable_name = variable_name
        self.compute = compute
        self.path = Path(self.name + ".wfn")

    def task_request_wave_function(self) -> Dict[str, Any]:
        db = load_db(self.db_path, self.variable_name)
        src_path = db.request(self.parameters, self.compute)

        @DoitAction
        def action_copy(targets):
            del targets

            dirname = self.path.parent
            if dirname and not dirname.exists():
                os.makedirs(dirname)

            shutil.copy2(src_path, self.path)

        return {
            "name": "wfn:{}:request".format(self.name),
            "actions": [action_copy],
            "targets": [self.path],
            "file_dep": [src_path],
            "verbosity": 2,
        }

    def task_request_wave_function_dry_run(self) -> Dict[str, Any]:
        db = load_db(self.db_path, self.variable_name)
        db.working_dir = self.db_path.parent / db.name

        with WorkingDir(self.db_path.parent):
            try:
                db.request(self.parameters, False)
            except MissingWfnError:
                pass

        @DoitAction
        def action_noop(targets):
            del targets
            pass

        return {
            "name": "wfn:{}:request_dry_run".format(self.name),
            "actions": [action_noop],
            "verbosity": 2,
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_request_wave_function]

    def get_tasks_dry_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_request_wave_function_dry_run]


class FrameFromPsi(Task):
    def __init__(
        self, psi: Union[str, Path], index: int, path: Union[Path, str] = None
    ):
        self.logger = get_logger(__name__ + ".FrameFromPsi")
        self.psi = make_path(psi)
        self.index = index

        if path is None:
            self.path = self.psi.stem + "_{}.wfn".format(self.index)
        else:
            self.path = make_path(path)

        self.pickle_path = self.path.with_suffix(".wfn_pickle")

    def task_pickle(self) -> Dict[str, Any]:
        @DoitAction
        def action_pickle(targets):
            with open(targets[0], "wb") as fptr:
                pickle.dump(self.index, fptr, protocol=3)

        return {
            "name": "frame_from_psi:{}_{}:pickle".format(self.path.stem, self.index),
            "actions": [action_pickle],
            "targets": [str(self.pickle_path)],
        }

    def task_grab(self) -> Dict[str, Any]:
        @DoitAction
        def action_grab(targets):
            with open(self.pickle_path, "rb") as fptr:
                index = pickle.load(fptr)

            tape, time, frame = read_psi_frame_ascii(self.psi, self.index)
            write_psi_ascii(
                targets[0], (tape, numpy.array([time]), numpy.array([frame]))
            )

        return {
            "name": "frame_from_psi:{}_{}:grab".format(self.path.stem, self.index),
            "actions": [action_grab],
            "file_dep": [self.pickle_path, self.psi],
            "targets": [self.path],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_pickle, self.task_grab]
