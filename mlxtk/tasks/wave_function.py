import os
import shutil
import sys
from typing import Any, Callable, Dict, List

from ..cwd import WorkingDir
from ..doit_compat import DoitAction
from ..log import get_logger
from ..parameters import Parameters
from ..wave_function_db import load_db
from .task import Task

LOGGER = get_logger(__name__)


class RequestWaveFunction(Task):
    def __init__(self,
                 name: str,
                 parameters: Parameters,
                 db_path: str,
                 variable_name: str = "db",
                 compute: bool = True):

        if not os.path.isabs(db_path):
            self.db_path = os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])), db_path)
        else:
            self.db_path = db_path

        self.name = name
        self.parameters = parameters
        self.variable_name = variable_name
        self.compute = compute
        self.path = self.name + ".wfn"

    def task_request_wave_function(self) -> Dict[str, Any]:
        db = load_db(self.db_path, self.variable_name)
        db.working_dir = os.path.join(os.path.dirname(self.db_path), db.name)

        with WorkingDir(os.path.dirname(self.db_path)):
            src_path = db.request(self.parameters, self.compute)

        @DoitAction
        def action_copy(targets):
            del targets

            dirname = os.path.dirname(self.path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            shutil.copy2(src_path, self.path)

        return {
            "name": "wfn:{}:request".format(self.name),
            "actions": [action_copy],
            "targets": [self.path],
            "file_dep": [src_path],
            "verbosity": 2
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_request_wave_function]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return []
