"""Work with a collection of simulations.
"""
import os
import sys
from pathlib import Path
from typing import List, Union

from mlxtk import sge
from mlxtk.log import get_logger
from mlxtk.simulation import Simulation
from mlxtk.util import make_path


class SimulationSetBase:
    """A collection of simulations.

    Args:
        name (str): name for this set of simulations
        simulations (list): the list of simulations
        working_dir (str): optional working directory

    Attributes:
        name (str): name for this set of simulations
        simulations (list): the list of simulations
        working_dir (str): optional working directory (defaults to ``name``)
    """

    def __init__(
        self,
        name: str,
        simulations: List[Simulation],
        working_dir: Union[str, Path] = None,
    ):
        self.name = name
        self.simulations = simulations
        self.working_dir = (
            Path(name).resolve() if working_dir is None else make_path(working_dir)
        )

        self.logger = get_logger(__name__ + ".SimulationSet")

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)
