from pathlib import Path
from typing import List, Union

from ..simulation import Simulation
from . import base


class SimulationSet(base.SimulationSetBase):
    def __init__(
            self,
            name: str,
            simulations: List[Simulation],
            working_dir: Union[str, Path] = None,
    ):
        super().__init__(name, simulations, working_dir)
