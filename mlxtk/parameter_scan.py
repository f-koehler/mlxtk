import os
import sys
from typing import Callable, Generator, Iterable, Optional

from . import parameters
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

    @staticmethod
    def format_parameters(combination: Parameters):
        return "_".join(
            [name + "=" + str(combination[name]) for name in combination.names]
        )

    def compute_simulations(self):
        self.simulations = []
        combinations = list(self.parameters)

        for combination in combinations:
            self.simulations.append(self.func(combination))
            self.simulations[-1].name = ParameterScan.format_parameters(combination)
            self.simulations[-1].working_dir = os.path.join(
                "sim", hash_string(self.simulations[-1].name)
            )
            self.simulations[-1].name = self.name + self.simulations[-1].name

    def main(self, args: Iterable[str] = sys.argv[1:]):
        self.compute_simulations()

        super().main(args)
