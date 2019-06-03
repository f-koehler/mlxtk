from abc import ABC, abstractmethod

import numpy

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

    def __init__(self, parameters: Parameters, grid: DVRSpecification):
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
            (self.grid_1b, ),
            {"kinetic_coeff": -0.5},
            {"kinetic": self.grid_1b.get_d2()},
            "kinetic_coeff | 1 kinetic",
        )

    def get_kinetic_operator(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, ),
            (self.grid, ),
            {"kinetic_coeff": -0.5},
            {
                "kinetic": {
                    "value": self.grid.get_d2(),
                    "fft": self.grid.is_fft()
                }
            },
            "kinetic_coeff | 1 kinetic",
        )

    def get_com_operator(self) -> MBOperatorSpecification:
        return MBOperatorSpecification((1, ), (self.grid, ), {"com_coeff": 1.},
                                       {"com": self.grid.get_x()},
                                       "com_coeff | 1 com")

    def get_com_operator_squared(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, ),
            (self.grid, ),
            {
                "com_squared_coeff_1": 1.0,
                "com_squared_coeff_2": 2.0,
            },
            {
                "com_squared_x": self.grid.get_x(),
                "com_squared_x^2": self.grid.get_x()**2,
            },
            [
                "com_squared_coeff_1 | 1 com_squared_x^2",
                "com_squared_coeff_2 | 1 com_squared_x | 1* com_squared_x",
            ],
        )

    def get_truncated_unit_operator(self, xmin: float,
                                    xmax: float) -> MBOperatorSpecification:
        term = numpy.zeros_like(self.grid.get_x())
        term[numpy.logical_and(self.grid.get_x() >= xmin,
                               self.grid.get_x() <= xmax)] = 1.
        return MBOperatorSpecification(
            (1, ), (self.grid, ), {"norm": 1. / self.parameters.N},
            {"truncated_one": term}, "norm | 1 truncated_one")

    def get_truncated_com_operator(self, xmin: float,
                                   xmax: float) -> MBOperatorSpecification:
        term = numpy.copy(self.grid.get_x())
        term[numpy.logical_or(term < xmin, term > xmax)] = 0.
        return MBOperatorSpecification((1, ), (self.grid, ), {"com_coeff": 1.},
                                       {"com": term}, "com_coeff | 1 com")

    def get_truncated_com_operator_squared(self, xmin: float, xmax: float
                                           ) -> MBOperatorSpecification:
        term = numpy.copy(self.grid.get_x())
        term[numpy.logical_or(term < xmin, term > xmax)] = 0.

        term2 = term**2
        return MBOperatorSpecification(
            (1, ),
            (self.grid, ),
            {
                "com_squared_coeff_1": 1.0,
                "com_squared_coeff_2": 2.0,
            },
            {
                "com_squared_x": term,
                "com_squared_x^2": term2,
            },
            [
                "com_squared_coeff_1 | 1 com_squared_x^2",
                "com_squared_coeff_2 | 1 com_squared_x | 1* com_squared_x",
            ],
        )
