import shutil
import os
import sys

from ..parameters import Parameters
from ..wave_function_db import load_db
from ..doit_compat import DoitAction
from ..log import get_logger
from ..cwd import WorkingDir

LOGGER = get_logger(__name__)


def request_wave_function(name: str,
                          parameters: Parameters,
                          db_path: str,
                          variable_name: str = "db",
                          compute: bool = True):
    if not os.path.isabs(db_path):
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), db_path)

    path = name + ".wfn"

    def task_request_wave_function():
        db = load_db(db_path, variable_name)
        db.working_dir = os.path.join(os.path.dirname(db_path), db.name)

        with WorkingDir(os.path.dirname(db_path)):
            src_path = db.request(parameters, compute)

        @DoitAction
        def action_copy(targets):
            del targets

            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            shutil.copy2(src_path, path)

        return {
            "name": "wfn:{}:request".format(name),
            "actions": [action_copy],
            "targets": [path],
            "file_dep": [src_path],
            "verbosity": 2
        }

    return [task_request_wave_function]
