import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from ..cwd import WorkingDir
from ..doit_compat import DoitAction
from ..log import get_logger
from ..parameters import Parameters
from ..util import make_path
from ..wave_function_db import MissingWfnError, load_db
from .task import Task

LOGGER = get_logger(__name__)


class RequestWaveFunction(Task):
    def __init__(self,
                 name: str,
                 parameters: Parameters,
                 db_path: Union[str, Path],
                 variable_name: str = "db",
                 compute: bool = True):

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
        db.working_dir = self.db_path.parent / db.name

        with WorkingDir(self.db_path.parent):
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
            "verbosity": 2
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
            "verbosity": 2
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_request_wave_function]

    def get_tasks_dry_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_request_wave_function_dry_run]
