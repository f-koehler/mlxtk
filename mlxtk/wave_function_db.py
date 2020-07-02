import argparse
import copy
import importlib.util
import pickle
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog

from mlxtk import cwd, doit_compat
from mlxtk.hashing import hash_string
from mlxtk.log import get_logger
from mlxtk.parameter_scan import ParameterScan
from mlxtk.parameters import Parameters
from mlxtk.simulation import Simulation
from mlxtk.util import get_main_path

assert List

LOGGER = get_logger(__name__)


class MissingWfnError(Exception):
    def __init__(self, parameters: Parameters):
        self.parameters = copy.deepcopy(parameters)
        super().__init__("Missing wave function for parameters: " + str(parameters))


class WaveFunctionDBLockError(Exception):
    def __init__(self, path: Path):
        super().__init__("Could not lock wave function db: " + str(path))


class WaveFunctionDBLock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        for _ in range(1024):
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
    def __init__(
        self,
        name: str,
        wfn_path: Path,
        prototype: Parameters,
        func: Callable[[Parameters], Simulation],
        working_dir: str = None,
    ):
        self.logger = get_logger(__name__ + ".WaveFunctionDB")
        self.prototype = prototype
        self.stored_wave_functions = []  # type: List[Parameters]
        self.missing_wave_functions = []  # type: List[Parameters]
        self.wfn_path = wfn_path

        super().__init__(name, func, [], working_dir)
        self.logger = get_logger(__name__ + ".WaveFunctionDB")

        self.argparser_tui = self.subparsers.add_parser("tui")
        self.argparser_tui.set_defaults(subcommand=self.cmd_tui)
        self.argparser_remove = self.subparsers.add_parser("remove")
        self.argparser_remove.set_defaults(subcommand=self.cmd_remove)
        self.argparser_remove.add_argument("selection", type=str)

        self.load_missing_wave_functions()
        self.load_stored_wave_functions()
        self.combinations = self.stored_wave_functions + self.missing_wave_functions

        # TODO: it would be better to collect missing wave functions and remove them in one go
        # this would require some additional methods
        for parameter in self.stored_wave_functions:
            path = self.get_path(parameter)
            if not path:
                self.store_missing_wave_function(parameter)
                self.remove_wave_function(parameter)
                continue

            if not path.exists():
                self.store_missing_wave_function(parameter)
                self.remove_wave_function(parameter)

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
                pickle.dump(entries, fptr, protocol=3)

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
                pickle.dump(entries, fptr, protocol=3)

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
                pickle.dump(entries, fptr, protocol=3)

            self.stored_wave_functions.append(parameters)

    def remove_wave_function(self, parameters: Parameters):
        self.create_working_dir()

        p = Path("stored_wave_functions.pickle")
        with cwd.WorkingDir(self.working_dir):
            if not p.exists():
                return

            with open(p, "rb") as fptr:
                entries = pickle.load(fptr)

            if parameters not in entries:
                return

            entries.remove(parameters)

            with open(p, "wb") as fptr:
                pickle.dump(entries, fptr, protocol=3)

            self.stored_wave_functions.remove(parameters)

    def get_simulation_path(self, parameters: Parameters) -> Optional[Path]:
        common_parameter_names = parameters.get_common_parameter_names(self.prototype)

        for p in self.stored_wave_functions:
            if p.has_same_common_parameters(parameters, common_parameter_names):
                path = self.working_dir.resolve() / "sim" / hash_string(repr(p))
                return path

        return None

    def get_path(self, parameters: Parameters) -> Optional[Path]:
        sim_path = self.get_simulation_path(parameters)
        if sim_path is None:
            return None

        return sim_path / self.wfn_path

    def request(self, parameters: Parameters, compute: bool = True) -> Path:
        self.logger.info("request wave function for parameters %s", repr(parameters))

        with cwd.WorkingDir(self.working_dir.parent):
            common_parameter_names = parameters.get_common_parameter_names(
                self.prototype
            )

            path = self.get_path(parameters)
            if path:
                self.logger.info("wave function is present")
                return path

            self.logger.info("wave function is not present")

            p = copy.deepcopy(self.prototype)
            for name in common_parameter_names:
                p[name] = parameters[name]

            if p not in self.missing_wave_functions:
                self.store_missing_wave_function(p)
                self.combinations = (
                    self.stored_wave_functions + self.missing_wave_functions
                )

            if not compute:
                raise MissingWfnError(p)

            self._compute(p)

            return self.request(parameters, compute)

    def _compute(self, parameters: Parameters):
        self.logger.info("computing wave function for parameters %s", repr(parameters))
        self.run_by_param(parameters)

        with WaveFunctionDBLock(self.working_dir / "wave_function_db.lock"):
            self.remove_missing_wave_function(parameters)
            self.store_wave_function(parameters)

    def cmd_run_index(self, args: argparse.Namespace):
        super().cmd_run_index(args)
        parameters = self.combinations[args.index]
        with WaveFunctionDBLock(self.working_dir / "wave_function_db.lock"):
            self.remove_missing_wave_function(parameters)
            self.store_wave_function(parameters)

    def cmd_run(self, args: argparse.Namespace):
        self.logger.info("running wave function db")

        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        self.create_working_dir()

        def task_run_simulation(parameters: Parameters, name: str) -> Dict[str, Any]:
            @doit_compat.DoitAction
            def action_run_simulation(targets: List[str]):
                del targets
                self.request(parameters, compute=True)

            return {
                "name": "run_simulation:" + name.replace("=", ":"),
                "actions": [action_run_simulation],
            }

        tasks = []  # type: List[Callable[[], Dict[str, Any]]]
        for simulation, parameter in zip(self.simulations, self.combinations):
            tasks += [partial(task_run_simulation, parameter, simulation.name)]

        doit_compat.run_doit(
            tasks,
            ["--process=" + str(args.jobs), "--backend=sqlite3", "--db-file=:memory:"],
        )

    def cmd_remove(self, args: argparse.Namespace):
        indices = self.parse_selection(args.selection)
        self.logger.info("remove wave_functions with indices: %s", str(indices))

        selected_parameters = [self.combinations[index] for index in indices]
        self.unlink_simulations()
        for parameters in selected_parameters:
            shutil.rmtree(self.get_simulation_path(parameters))
            self.remove_missing_wave_function(parameters)
            self.remove_wave_function(parameters)
            i = self.combinations.index(parameters)
            self.combinations.pop(i)
            self.simulations.pop(i)

        self.store_parameters()
        self.link_simulations()

    def cmd_tui(self, args: argparse.Namespace):
        self.logger.info("start terminal ui")
        selection = checkboxlist_dialog(
            title="Select simulations",
            text="Select simulations to run operations on.",
            values=[
                (i, repr(combination))
                for i, combination in enumerate(self.combinations)
            ],
        ).run()

        if not selection:
            self.logger.info("no simulations selected, nothing to do")
            return

        self.logger.info("selected %d simulation(s)", len(selection))

        operation = radiolist_dialog(
            title="Select operation",
            text="Select operation to run on selected simulations",
            values=[("remove", "remove")],
        ).run()

        if not operation:
            self.logger.info("no operation selected, nothing to do")
            return

        self.logger.info(
            'running operation "%s" on %d simulation(s)', operation, len(selection)
        )

        if operation == "remove":
            cmd = [
                sys.executable,
                Path(get_main_path()).resolve(),
                "remove",
                ",".join((str(i) for i in selection)),
            ]
            self.logger.info("command: %s", str(cmd))
            subprocess.run(cmd)


def load_db(path: Path, variable_name: str = "db") -> WaveFunctionDB:
    with cwd.WorkingDir(path.resolve().parent):
        spec = importlib.util.spec_from_file_location("db", str(path))
        db_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(db_module)
        db = getattr(db_module, variable_name)

    db.working_dir = path.parent / db.name
    return db
