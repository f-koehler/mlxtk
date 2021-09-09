"""Scan over ranges of parameters

This module provides facilities to create and run sets of simulations based on
parameter ranges.
"""
import argparse
from multiprocessing.sharedctypes import Value

from pytest import param
from itertools import combinations
import os
import pickle
import subprocess
from pathlib import Path
from typing import Callable, List, Union
import sys

import mlxtk.parameters
from mlxtk import cwd
from mlxtk.hashing import hash_string
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.simulation import Simulation
from mlxtk.simulation_set import SimulationSet
from mlxtk.cwd import WorkingDir

assert List


class ParameterScan(SimulationSet):
    def __init__(
        self,
        name: str,
        func: Callable[[Parameters], Simulation],
        combinations: List[Parameters],
        working_dir: Union[str, Path] = None,
    ):
        super().__init__(name, [], working_dir)
        self.func = func
        self.logger = get_logger(__name__ + ".ParameterScan")
        self.simulations = []  # type: List[Simulation]
        self.combinations = combinations

    def compute_simulation_name(self, parameters) -> str:
        return f"{self.name}_{repr(parameters)}"

    def compute_working_dir(self, parameters: Parameters) -> Path:
        return self.working_dir / "sim" / hash_string(repr(parameters))

    def compute_simulation(self, parameters: Parameters) -> Simulation:
        simulation = self.func(parameters)
        simulation.name = self.compute_simulation_name(parameters)
        simulation.working_dir = self.compute_working_dir(parameters)
        return simulation

    def compute_simulations(self):
        self.simulations = []

        self.simulations = [
            self.compute_simulation(parameters) for parameters in self.combinations
        ]

    def store_parameters(self):
        self.logger.info("storing scan parameters")
        self.create_working_dir()

        if not self.combinations:
            return

        with cwd.WorkingDir(self.working_dir):
            for combination in self.combinations:
                working_dir = self.compute_working_dir(combination)
                working_dir.mkdir(exist_ok=True, parents=True)
                with cwd.WorkingDir(working_dir):
                    with open("parameters.pickle", "wb") as fptr:
                        pickle.dump(combination, fptr, protocol=3)

                    with open("parameters.json", "w") as fptr:
                        fptr.write(combination.to_json() + "\n")

            with open("scan.pickle", "wb") as fptr:
                pickle.dump(
                    [combination for combination in self.combinations], fptr, protocol=3
                )

    def link_simulations(self):
        if not self.combinations:
            return

        self.logger.info("create simulation symlinks")
        self.create_working_dir()

        # link by index
        with cwd.WorkingDir(self.working_dir):
            path_by_index = Path("by_index")
            if not path_by_index.exists():
                os.makedirs(path_by_index)

            for index, combination in enumerate(self.combinations):
                path = self.compute_working_dir(combination)
                link = path_by_index / str(index)

                if link.exists():
                    link.unlink()
                subprocess.run(["ln", "-s", "-f", "-r", path, link])

        # link by parameters
        with cwd.WorkingDir(self.working_dir):
            path_by_param = Path("by_param")
            if not path_by_param.exists():
                os.makedirs(path_by_param)

            variables, constants = mlxtk.parameters.get_variables(self.combinations)
            for combination in self.combinations:
                try:
                    if not variables:
                        name = "_".join(
                            constant + "=" + str(combination[constant])
                            for constant in constants
                        )
                    else:
                        name = "_".join(
                            variable + "=" + str(combination[variable])
                            for variable in variables
                        )
                    path = self.compute_working_dir(combination)
                    link = path_by_param / name

                    if link.exists():
                        link.unlink()
                    subprocess.run(["ln", "-s", "-f", "-r", path, link])
                except OSError:
                    continue

    def unlink_simulations(self):
        if not self.working_dir.exists():
            return

        with cwd.WorkingDir(self.working_dir):
            path_by_index = Path("by_index")
            if path_by_index.exists():
                with cwd.WorkingDir(path_by_index):
                    for entry in os.listdir(os.getcwd()):
                        path_entry = Path(entry)
                        if path_entry.is_symlink():
                            path_entry.unlink()

            path_by_param = Path("by_param")
            if path_by_param.exists():
                with cwd.WorkingDir(path_by_param):
                    for entry in os.listdir(os.getcwd()):
                        path_entry = Path(entry)
                        if path_entry.is_symlink():
                            path_entry.unlink()

    def cmd_clean(self, args: argparse.Namespace):
        self.compute_simulations()
        super().cmd_clean(args)

    def cmd_dry_run(self, args: argparse.Namespace):
        self.compute_simulations()
        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        super().cmd_dry_run(args)

    def cmd_list(self, args: argparse.Namespace):
        if args.directory:
            for i, combination in enumerate(self.combinations):
                print(i, self.compute_working_dir(combination))
        else:
            for i, combination in enumerate(self.combinations):
                print(i, self.compute_simulation_name(combination))

    def cmd_list_tasks(self, args: argparse.Namespace):
        self.compute_simulation(args.index).main(["list"])

    def cmd_propagation_status(self, args: argparse.Namespace):
        self.compute_simulations()
        super().cmd_propagation_status(args)

    def cmd_qdel(self, args: argparse.Namespace):
        self.compute_simulations()
        super().cmd_qdel(args)

    def cmd_qsub_array(self, args: argparse.Namespace):
        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        super().cmd_qsub_array(args)

    def cmd_run(self, args: argparse.Namespace):
        self.compute_simulations()
        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        super().cmd_run(args)

    def cmd_run_index(self, args: argparse.Namespace):
        script_dir = Path(sys.argv[0]).parent.resolve()
        with WorkingDir(script_dir):
            self.logger.info("run simulation with index %d", args.index)
            self.compute_simulation(self.combinations[args.index]).main(["run"])

    def run_by_param(self, parameters: mlxtk.parameters.Parameters):
        self.logger.info("run simulation for parameters %s", repr(parameters))

        try:
            index = combinations.index(parameters)
            simulation = self.compute_simulation(index)
            simulation.main(["run-index", str(index)])
        except ValueError:
            raise ValueError("Parameters not included: " + str(parameters))

    def cmd_task_info(self, args: argparse.Namespace):
        self.create_working_dir()

        with WorkingDir(self.working_dir):
            self.compute_simulation(self.combinations[args.index]).main(
                ["task-info", args.name]
            )

    def main(self, argv: List[str] = None):
        # self.compute_simulations()

        super().main(argv)
