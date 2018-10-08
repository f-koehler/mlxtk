from typing import Callable, Generator, Iterable, Optional

from .parameters import Parameters
from .simulation import Simulation
from .simulation_set import SimulationSet
from .log import get_logger


class ParameterScan(SimulationSet):
    def __init__(
        self,
        name: str,
        func: Callable[[Parameters], Simulation],
        parameters: Generator[Parameters, None, None],
        working_dir: Optional[str] = None,
    ):
        super(self, ParameterScan).__init__(name, [], working_dir)
        self.func = func
        self.parameters = parameters
        self.logger = get_logger(__name__)

    def compute_simulations(self):
        self.simulations = [self.func(combination) for combination in self.parameters]

    def main(self, args: Iterable[str] = sys.argv[1:]):
        self.compute_simulations()

        super(self, ParameterScan).main(args)
