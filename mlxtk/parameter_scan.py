"""Scan over ranges of parameters

This module provides facilities to create and run sets of simulations based on
parameter ranges.
"""
import argparse
import os
import pickle
import subprocess
from pathlib import Path
from typing import Callable, List, Union

import mlxtk.parameters
from mlxtk import cwd
from mlxtk.hashing import hash_string
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.simulation import Simulation
from mlxtk.simulation_set import SimulationSet

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

    def compute_simulations(self):
        self.simulations = []

        for combination in self.combinations:
            self.simulations.append(self.func(combination))
            self.simulations[-1].name = repr(combination)
            self.simulations[-1].working_dir = (
                self.working_dir / "sim" / hash_string(self.simulations[-1].name)
            )
            self.simulations[-1].name = self.name + "_" + self.simulations[-1].name

    def store_parameters(self):
        self.logger.info("storing scan parameters")
        self.create_working_dir()

        if not self.combinations:
            return

        with cwd.WorkingDir(self.working_dir):
            for combination, simulation in zip(self.combinations, self.simulations):
                simulation.create_working_dir()
                with cwd.WorkingDir(simulation.working_dir):
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

            for index, simulation in enumerate(self.simulations):
                path = simulation.working_dir
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
            for combination, simulation in zip(self.combinations, self.simulations):
                if not variables:
                    name = "_".join(
                        (
                            constant + "=" + str(combination[constant])
                            for constant in constants
                        )
                    )
                else:
                    name = "_".join(
                        (
                            variable + "=" + str(combination[variable])
                            for variable in variables
                        )
                    )
                path = simulation.working_dir
                link = path_by_param / name

                if link.exists():
                    link.unlink()
                subprocess.run(["ln", "-s", "-f", "-r", path, link])

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

    def cmd_dry_run(self, args: argparse.Namespace):
        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        super().cmd_dry_run(args)

    def cmd_run(self, args: argparse.Namespace):
        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        super().cmd_run(args)

    def run_by_param(self, parameters: mlxtk.parameters.Parameters):
        self.logger.info("run simulation for parameters %s", repr(parameters))
        self.compute_simulations()
        for i, combination in enumerate(self.combinations):
            if combination != parameters:
                continue

            self.main(["run-index", str(i)])
            return

        raise ValueError("Parameters not included: " + str(parameters))

    def cmd_qsub_array(self, args: argparse.Namespace):
        self.unlink_simulations()
        self.store_parameters()
        self.link_simulations()

        super().cmd_qsub_array(args)

    def main(self, argv: List[str] = None):
        self.compute_simulations()

        super().main(argv)
