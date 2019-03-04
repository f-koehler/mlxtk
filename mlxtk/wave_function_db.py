import copy
import importlib.util
import os.path
import pickle
from typing import Callable, Generator

from .parameters import Parameters
from .parameter_scan import ParameterScan
from . import cwd
from .hashing import hash_string
from .simulation import Simulation


class MissingWfnError(Exception):
    def __init__(self, parameters):
        self.parameters = copy.deepcopy(parameters)
        super().__init__("Missing wave function for parameters: " +
                         str(parameters))


class WaveFunctionDB(ParameterScan):
    def __init__(self,
                 name: str,
                 wfn_path: str,
                 prototype: Parameters,
                 func: Callable[[Parameters], Simulation],
                 working_dir: str = None):
        self.prototype = prototype
        self.stored_wave_functions = []
        self.missing_wave_functions = []
        self.wfn_path = wfn_path

        def generator() -> Generator[Parameters, None, None]:
            for p in self.stored_wave_functions:
                yield p

            for p in self.missing_wave_functions:
                yield p

        super().__init__(name, func, generator(), working_dir)

        self.load_missing_wave_functions()
        self.load_stored_wave_functions()

    def load_missing_wave_functions(self):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            if not os.path.exists("missing_wave_functions.pickle"):
                return

            with open("missing_wave_functions.pickle", "rb") as fp:
                self.missing_wave_functions = pickle.load(fp)

    def load_stored_wave_functions(self):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            if not os.path.exists("scan.pickle"):
                return

            with open("scan.pickle", "rb") as fp:
                self.stored_wave_functions = pickle.load(fp)

    def store_missing_wave_functions(self):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            with open("missing_wave_functions.pickle", "wb") as fp:
                pickle.dump(self.missing_wave_functions, fp)

    def request(self, parameters: Parameters):
        common_parameters = parameters.get_common_parameters(self.prototype)

        for p in self.stored_wave_functions:
            if p.has_same_common_parameters(parameters, common_parameters):
                path = os.path.join(
                    os.path.abspath(self.working_dir), "sim",
                    hash_string(p.name), self.wfn_path)
                return path

        p = copy.deepcopy(self.prototype)
        for name in common_parameters:
            p[name] = parameters[name]

        if p not in self.missing_wave_functions:
            self.missing_wave_functions.append(p)
            self.store_missing_wave_functions()
        raise MissingStateError(p)


def load_db(path: str, variable_name: str = "db") -> WaveFunctionDB:
    spec = importlib.util.spec_from_file_location("db_module", path)
    db_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db_module)
    db = getattr(db_module, variable_name)
    db.working_dir = os.path.abspath(os.path.dirname(path))
    return db
