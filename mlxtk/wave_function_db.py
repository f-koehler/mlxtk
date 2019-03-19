import copy
import importlib.util
import os.path
import pathlib
import pickle
import time
from typing import Callable

from . import cwd
from .hashing import hash_string
from .log import get_logger
from .parameter_scan import ParameterScan
from .parameters import Parameters
from .simulation import Simulation

LOGGER = get_logger(__name__)


class MissingWfnError(Exception):
    def __init__(self, parameters: Parameters):
        self.parameters = copy.deepcopy(parameters)
        super().__init__("Missing wave function for parameters: " +
                         str(parameters))


class WaveFunctionDBLockError(Exception):
    def __init__(self, path: str):
        super().__init__("Could not lock wave function db: " + path)


class WaveFunctionDBLock:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        p = pathlib.Path(self.path)
        for i in range(1024):
            try:
                p.touch(exist_ok=False)
                return
            except FileExistsError:
                time.sleep(1)
                continue
        raise WaveFunctionDBLockError(self.path)

    def __exit__(self, _type, value, traceback):
        del _type
        del value
        del traceback
        os.remove(self.path)


class WaveFunctionDB(ParameterScan):
    def __init__(self,
                 name: str,
                 wfn_path: str,
                 prototype: Parameters,
                 func: Callable[[Parameters], Simulation],
                 working_dir: str = None):
        self.logger = get_logger(__name__)
        self.prototype = prototype
        self.stored_wave_functions = []
        self.missing_wave_functions = []
        self.wfn_path = wfn_path

        super().__init__(name, func, [], working_dir)
        self.logger = get_logger(__name__)

        self.load_missing_wave_functions()
        self.load_stored_wave_functions()
        self.combinations = self.stored_wave_functions + self.missing_wave_functions

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
            if not os.path.exists("stored_wave_functions.pickle"):
                return

            with open("stored_wave_functions.pickle", "rb") as fp:
                self.stored_wave_functions = pickle.load(fp)

    def store_missing_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            entries = []
            if os.path.exists("missing_wave_functions.pickle"):
                with open("missing_wave_functions.pickle", "rb") as fp:
                    entries = pickle.load(fp)

            if parameters in entries:
                return

            entries.append(parameters)

            with open("missing_wave_functions.pickle", "wb") as fp:
                pickle.dump(entries, fp)

    def remove_missing_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            if not os.path.exists("missing_wave_functions.pickle"):
                return

            with open("missing_wave_functions.pickle", "rb") as fp:
                entries = pickle.load(fp)

            if parameters not in entries:
                return

            entries.remove(parameters)

            with open("missing_wave_functions.pickle", "wb") as fp:
                pickle.dump(entries, fp)

    def store_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        with cwd.WorkingDir(self.working_dir):
            entries = []
            if os.path.exists("stored_wave_functions.pickle"):
                with open("stored_wave_functions.pickle", "rb") as fp:
                    entries = pickle.load(fp)

            if parameters in entries:
                return

            entries.append(parameters)

            with open("stored_wave_functions.pickle", "wb") as fp:
                pickle.dump(entries, fp)

    def request(self, parameters: Parameters, compute: bool = True):
        self.logger.info("request wave function for parameters %s",
                         repr(parameters))
        common_parameter_names = parameters.get_common_parameter_names(
            self.prototype)

        for p in self.stored_wave_functions:
            if p.has_same_common_parameters(parameters,
                                            common_parameter_names):
                self.logger.info("wave function is present")
                path = os.path.join(
                    os.path.abspath(self.working_dir), "sim",
                    hash_string(repr(p)), self.wfn_path)
                return path

        p = copy.deepcopy(self.prototype)
        for name in common_parameter_names:
            p[name] = parameters[name]

        if p not in self.missing_wave_functions:
            self.missing_wave_functions.append(p)
            self.store_missing_wave_function(p)
            self.combinations = self.stored_wave_functions + self.missing_wave_functions

        if not compute:
            self.logger.error(
                "wave function is not present and computation is disabled")
            raise MissingWfnError(p)

        self._compute(p)

        return self.request(parameters, compute)

    def _compute(self, parameters: Parameters):
        self.logger.info("computing wave function for parameters %s",
                         repr(parameters))
        self.run_by_param(parameters)

        with WaveFunctionDBLock(
                os.path.join(self.working_dir, "wave_function_db.lock")):
            self.remove_missing_wave_function(parameters)
            self.store_wave_function(parameters)
            self.stored_wave_functions.append(parameters)
            self.missing_wave_functions.remove(parameters)


def load_db(path: str, variable_name: str = "db") -> WaveFunctionDB:
    spec = importlib.util.spec_from_file_location("db", path)
    db_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db_module)
    db = getattr(db_module, variable_name)
    return db
