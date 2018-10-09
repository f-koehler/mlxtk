from mlxtk import dvr, tasks
from mlxtk.parameters import Parameters


class HarmonicTrap(object):
    """A class for a quantum harmonic oscillator with interacting particles

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
        grid: dvr.DVRSpecification = dvr.add_harmdvr(400, 0., 1.),
    ):
        self.parameters = parameters
        self.grid = grid

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("g", 0.1, "strength of the contact interaction between the particles"),
                ("m", 5, "number of single particle functions"),
                ("mass", 1.0, "mass of the particles"),
                ("omega", 1.0, "angular frequency of the harmonic trap"),
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
            {
                "kin": -1. / (2. * self.parameters.mass),
                "pot": 0.5 * self.parameters.mass * (self.parameters.omega ** 2),
            },
            {"dx²": self.grid.get_d2(), "x²": self.grid.get_x() ** 2},
            ["kin | 1 dx²", "pot | 1 x²"],
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

        coefficients = {
            "kin": -1. / (2. * self.parameters.mass),
            "pot": 0.5 * self.parameters.mass * (self.parameters.omega ** 2),
        }
        terms = {"dx²": self.grid.get_d2(), "x²": self.grid.get_x() ** 2}
        table = ["kin | 1 dx²", "pot | 1 x²"]

        if self.parameters.g != 0.0:
            coefficients["int"] = self.parameters.g
            terms["delta"] = self.grid.get_delta()
            table.append("int | {1:1} delta")

        return tasks.create_many_body_operator(
            name, (1,), (self.grid,), coefficients, terms, table
        )
