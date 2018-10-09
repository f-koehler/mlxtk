import numpy
from typing import Union

from mlxtk import dvr, tasks
from mlxtk.parameters import Parameters


def gaussian(x: Union[float, numpy.ndarray], V0: float, µ: float, w: float):
    return V0 * numpy.exp(-0.5 * (((x - µ) / w) ** 2))


class GaussianTrap(object):
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
        grid: dvr.DVRSpecification = dvr.add_harmdvr(400, 0., 1.),
    ):
        self.parameters = parameters
        self.grid = grid

    @staticmethod
    def create_parameters():
        return Parameters(
            [
                ["V0", 6.0, "depth of the Gaussian well"],
                ["µ", 1.0, "center of the Gaussian well"],
                ["w", 2.0, "width of the Gaussian well"],
                ["g", 0.1, "strength of the contact interaction"],
            ]
        )

    def get_1b_hamiltonian(self, name: str):
        """Create the single-body Hamiltonian
        Args:
           name (str): name for the operator file

        Returns:
           list: list of task dictionaries to create this operator
        """
        return tasks.create_operator(
            name,
            (self.grid,),
            {"kin": -1., "pot": -1.},
            {
                "dx²": self.grid.get_d2(),
                "gaussian": gaussian(
                    self.grid.get_x(),
                    self.parameters.V0,
                    self.parameters.µ,
                    self.parameters.w,
                ),
            },
            ["kin | 1 dx²", "pot | 1 gaussian"],
        )

    def get_mb_hamiltonian(self, name: str):
        """Create the many body hamiltonian

        The system parameters can be changed for this operator.
        This is useful when quenching a system parameter.

        Args:
           name (str): name for the operator file

        Returns:
           list: list of task dictionaries to create this operator
        """

        coefficients = {"kin": -1., "pot": -1.}
        terms = {
            "dx²": self.grid.get_d2(),
            "gaussian": gaussian(
                self.grid.get_x(),
                self.parameters.V0,
                self.parameters.µ,
                self.parameters.w,
            ),
        }
        table = ["kin | 1 dx²", "pot | 1 gaussian"]

        if self.parameters.g != 0.0:
            coefficients["int"] = self.parameters.g
            terms["delta"] = self.grid.get_delta()
            table.append("int | {1:1} delta")

        return tasks.create_many_body_operator(
            name, (1,), (self.grid,), coefficients, terms, table
        )