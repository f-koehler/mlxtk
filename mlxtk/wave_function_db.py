import argparse
import copy
import importlib.util
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List

from . import cwd, doit_compat
from .hashing import hash_string
from .log import get_logger
from .parameter_scan import ParameterScan
from .parameters import Parameters
from .simulation import Simulation

assert List

LOGGER = get_logger(__name__)


class MissingWfnError(Exception):
    def __init__(self, parameters: Parameters):
        self.parameters = copy.deepcopy(parameters)
        super().__init__("Missing wave function for parameters: " +
                         str(parameters))


class WaveFunctionDBLockError(Exception):
    def __init__(self, path: Path):
        super().__init__("Could not lock wave function db: " + str(path))


class WaveFunctionDBLock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        for i in range(1024):
            try:
                self.path.touch(exist_ok=False)
                return
            except FileExistsError:
                time.sleep(1)
                continue
        raise WaveFunctionDBLockError(self.path)

    def __exit__(self, _type, value, traceback):
        del _type
        del value
        del traceback
        self.path.unlink()


class WaveFunctionDB(ParameterScan):
    def __init__(self,
                 name: str,
                 wfn_path: Path,
                 prototype: Parameters,
                 func: Callable[[Parameters], Simulation],
                 working_dir: str = None):
        self.logger = get_logger(__name__ + ".WaveFunctionDB")
        self.prototype = prototype
        self.stored_wave_functions = []  # type: List[Parameters]
        self.missing_wave_functions = []  # type: List[Parameters]
        self.wfn_path = wfn_path

        super().__init__(name, func, [], working_dir)
        self.logger = get_logger(__name__ + ".WaveFunctionDB")

        self.load_missing_wave_functions()
        self.load_stored_wave_functions()
        self.combinations = self.stored_wave_functions + self.missing_wave_functions

    def load_missing_wave_functions(self):
        self.create_working_dir()

        p = Path("missing_wave_functions.pickle")
        with cwd.WorkingDir(self.working_dir):
            if not p.exists():
                return

            with open("missing_wave_functions.pickle", "rb") as fptr:
                self.missing_wave_functions = pickle.load(fptr)

    def load_stored_wave_functions(self):
        self.create_working_dir()

        p = Path("stored_wave_functions.pickle")
        with cwd.WorkingDir(self.working_dir):
            if not p.exists():
                return

            with open("stored_wave_functions.pickle", "rb") as fptr:
                self.stored_wave_functions = pickle.load(fptr)

    def store_missing_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        p = Path("missing_wave_functions.pickle")
        with cwd.WorkingDir(self.working_dir):
            entries = []  # type: List[Parameters]
            if p.exists():
                with open(p, "rb") as fptr:
                    entries = pickle.load(fptr)

            if parameters in entries:
                return

            entries.append(parameters)

            with open(p, "wb") as fptr:
                pickle.dump(entries, fptr)

            self.missing_wave_functions.append(parameters)

    def remove_missing_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        p = Path("missing_wave_functions.pickle")
        with cwd.WorkingDir(self.working_dir):
            if not p.exists():
                return

            with open(p, "rb") as fptr:
                entries = pickle.load(fptr)

            if parameters not in entries:
                return

            entries.remove(parameters)

            with open(p, "wb") as fptr:
                pickle.dump(entries, fptr)

            self.missing_wave_functions.remove(parameters)

    def store_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        p = Path("stored_wave_functions.pickle")
        with cwd.WorkingDir(self.working_dir):
            entries = []  # type: List[Parameters]
            if p.exists():
                with open(p, "rb") as fptr:
                    entries = pickle.load(fptr)

            if parameters in entries:
                return

            entries.append(parameters)

            with open(p, "wb") as fptr:
                pickle.dump(entries, fptr)

            self.stored_wave_functions.append(parameters)

    def request(self, parameters: Parameters, compute: bool = True) -> Path:
        self.logger.info("request wave function for parameters %s",
                         repr(parameters))

        common_parameter_names = parameters.get_common_parameter_names(
            self.prototype)

        for p in self.stored_wave_functions:
            if p.has_same_common_parameters(parameters,
                                            common_parameter_names):
                self.logger.info("wave function is present")
                path = self.working_dir.resolve() / "sim" / hash_string(
                    repr(p)) / self.wfn_path
                return path

        self.logger.info("wave function is not present")

        p = copy.deepcopy(self.prototype)
        for name in common_parameter_names:
            p[name] = parameters[name]

        if p not in self.missing_wave_functions:
            self.store_missing_wave_function(p)
            self.combinations = self.stored_wave_functions + self.missing_wave_functions

        if not compute:
            raise MissingWfnError(p)

        self._compute(p)

        return self.request(parameters, compute)

    def _compute(self, parameters: Parameters):
        self.logger.info("computing wave function for parameters %s",
                         repr(parameters))
        self.run_by_param(parameters)

        with WaveFunctionDBLock(self.working_dir / "wave_function_db.lock"):
            self.remove_missing_wave_function(parameters)
            self.store_wave_function(parameters)

    def run_index(self, args: argparse.Namespace):
        super().run_index(args)
        parameters = self.combinations[args.index]
        with WaveFunctionDBLock(self.working_dir / "wave_function_db.lock"):
            self.remove_missing_wave_function(parameters)
            self.store_wave_function(parameters)

    def run(self, args: argparse.Namespace):
        self.logger.info("running wave function db")

        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        self.create_working_dir()

        def task_run_simulation(parameters: Parameters,
                                name: str) -> Dict[str, Any]:
            @doit_compat.DoitAction
            def action_run_simulation(targets: List[str]):
                del targets
                self.request(parameters, compute=True)

            return {
                "name": "run_simulation:" + name.replace("=", ":"),
                "actions": [action_run_simulation]
            }

        tasks = []  # type: List[Callable[[], Dict[str, Any]]]
        for simulation, parameter in zip(self.simulations, self.combinations):
            tasks += [partial(task_run_simulation, parameter, simulation.name)]

        doit_compat.run_doit(tasks, [
            "--process=" + str(args.jobs), "--backend=sqlite3",
            "--db-file=" + str(self.working_dir / "doit.sqlite3")
        ])


def load_db(path: Path, variable_name: str = "db") -> WaveFunctionDB:
    with cwd.WorkingDir(path.parent):
        spec = importlib.util.spec_from_file_location("db", str(path))
        db_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(db_module)
        db = getattr(db_module, variable_name)
        return db
