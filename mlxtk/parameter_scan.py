import argparse
import os
import pickle
import subprocess
import sys
from typing import Callable, Generator, Iterable, Optional

from . import cwd, parameters
from .hashing import hash_string
from .log import get_logger
from .parameters import Parameters
from .simulation import Simulation
from .simulation_set import SimulationSet


class ParameterScan(SimulationSet):
    def __init__(
        self,
        name: str,
        func: Callable[[Parameters], Simulation],
        parameters: Generator[Parameters, None, None],
        working_dir: Optional[str] = None,
    ):
        super().__init__(name, [], working_dir)
        self.func = func
        self.parameters = parameters
        self.logger = get_logger(__name__)
        self.simulations = []  # type: List[Simulation]
        self.combinations = []  # type: List[Parameters]

    def compute_simulations(self):
        self.simulations = []
        self.combinations = list(self.parameters)

        for combination in self.combinations:
            self.simulations.append(self.func(combination))
            self.simulations[-1].name = repr(combination)
            self.simulations[-1].working_dir = os.path.join(
                "sim", hash_string(self.simulations[-1].name)
            )
            self.simulations[-1].name = self.name + self.simulations[-1].name

    def store_parameters(self):
        self.create_working_dir()

        cwd.change_dir(self.working_dir)
        for combination, simulation in zip(self.combinations, self.simulations):
            simulation.create_working_dir()
            cwd.change_dir(simulation.working_dir)
            with open("parameters.pickle", "wb") as fp:
                pickle.dump(combination, fp)

            with open("parameters.json", "w") as fp:
                fp.write(combination.to_json() + "\n")
            cwd.go_back()
        cwd.go_back()

    def link_simulations(self):
        self.create_working_dir()

        cwd.change_dir(self.working_dir)
        if not os.path.exists("by_param"):
            os.makedirs("by_param")

        variables, _ = parameters.get_variables(self.combinations)
        for combination, simulation in zip(self.combinations, self.simulations):
            name = "_".join(
                (variable + "=" + str(combination[variable]) for variable in variables)
            )
            path = simulation.working_dir
            link = os.path.join("by_param", name)

            subprocess.run(["ln", "-s", "-f", "-r", path, link])

        cwd.go_back()

    def run(self, args: argparse.Namespace):
        self.store_parameters()
        self.link_simulations()

        super().run(args)

    def qsub(self, args: argparse.Namespace):
        self.store_parameters()
        self.link_simulations()

        super().qsub(args)

    def main(self, args: Iterable[str] = sys.argv[1:]):
        self.compute_simulations()

        super().main(args)
