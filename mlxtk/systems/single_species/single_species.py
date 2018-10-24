from abc import ABC, abstractmethod

from mlxtk.dvr import DVRSpecification
from mlxtk.parameters import Parameters
from mlxtk.tasks import MBOperatorSpecification, OperatorSpecification


class SingleSpeciesSystem(ABC):
    """An abstract base class for single species systems

    Args:
        parameters (mlxtk.parameters.Parameters): the system parameters
        grid (mlxtk.dvr.DVRSpecification): the grid of the harmonic oscillator

    Attributes:
        parameters (mlxtk.parameters.Parameters): the system parameters
        grid (mlxtk.dvr.DVRSpecification): the grid of the harmonic oscillator
    """

    def __init__(
        self,
        parameters: Parameters,
        grid: DVRSpecification,
    ):
        self.parameters = parameters
        self.grid = grid

        if grid.is_fft():
            self.grid_1b = grid.get_expdvr()
        else:
            self.grid_1b = self.grid

    @staticmethod
    @abstractmethod
    def create_parameters() -> Parameters:
        pass

    def get_kinetic_operator_1b(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"kinetic_coeff": -0.5},
            {"kinetic": self.grid_1b.get_d2()},
            "kinetic_coeff | 1 kinetic",
        )

    def get_kinetic_operator(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"kinetic_coeff": -0.5},
            {"kinetic": {"value": self.grid.get_d2(), "fft": self.grid.is_fft()}},
            "kinetic_coeff | 1 kinetic",
        )
