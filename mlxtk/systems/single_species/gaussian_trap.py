from typing import Union

import numpy

from mlxtk import dvr, tasks
from mlxtk.parameters import Parameters


def gaussian(x: Union[float, numpy.ndarray], x0: float):
    return numpy.exp(-((x - x0) ** 2))


class GaussianTrap:
    """Interacting particles in a Gaussian shape trap

    Args:
        parameters (mlxtk.parameters.Parameters): the system parameters
        grid (mlxtk.dvr.DVRSpecification): the grid of the system

    Attributes:
        parameters (mlxtk.parameters.Parameters): the system parameters
        grid (mlxtk.dvr.DVRSpecification): the grid of the system
    """

    def __init__(
        self,
        parameters: Parameters,
        grid: dvr.DVRSpecification = dvr.add_harmdvr(400, 0.0, 1.0),
    ):
        self.parameters = parameters
        self.grid = grid

    @staticmethod
    def create_parameters():
        return Parameters(
            [
                ("V0", 1.0, "depth of the Gaussian well"),
                ("x0", 0.0, "center of the Gaussian well"),
                ("g", 0.1, "strength of the contact interaction"),
            ]
        )

    def get_kinetic_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid,),
            {"kinetic_coeff": -0.5},
            {"kinetic": self.grid.get_d2()},
            ["kinetic_coeff | 1 kinetic"],
        )

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid,),
            {"potential_coeff": -self.parameters.V0},
            {"potential": gaussian(self.grid.get_x(), self.grid.get_x())},
            ["potential_coeff | 1 potential"],
        )

    def get_1b_hamiltonian(self) -> tasks.OperatorSpecification:
        return self.get_kinetic_operator_1b() + self.get_potential_operator_1b()

    def get_mb_hamiltonian(self, name: str):
        """Create the many body hamiltonian

        The system parameters can be changed for this operator.
        This is useful when quenching a system parameter.

        Args:
           name (str): name for the operator file

        Returns:
           list: list of task dictionaries to create this operator
        """

        coefficients = {"kin": -0.5, "pot": -self.parameters.V0}
        terms = {
            "dx²": self.grid.get_d2(),
            "gaussian": gaussian(self.grid.get_x(), self.parameters.x0),
        }
        table = ["kin | 1 dx²", "pot | 1 gaussian"]

        if self.parameters.g != 0.0:
            coefficients["int"] = self.parameters.g
            terms["delta"] = self.grid.get_delta()
            table.append("int | {1:1} delta")

        return tasks.create_many_body_operator(
            name, (1,), (self.grid,), coefficients, terms, table
        )
