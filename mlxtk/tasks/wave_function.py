from ..parameters import Parameters
from ..wave_function_db import load_db, Wave


def request_wave_function(name: str, parameters: Parameters, db: str,
                          variable_name: str):
    def task_request_wave_function():
        path = name + ".wfn"

        db = load_db(path, variable_name)
        try:
            src_path = db.request(parameters)
        except MissingWfnError:
            pass

        return {
            "name": "wfn:{}:request".format(name),
            "actions": [],
            "targets": [path]
            "file_dep": [src_path]
        }
