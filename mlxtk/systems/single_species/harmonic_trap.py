from mlxtk import dvr, tasks
from mlxtk.parameters import Parameters


class HarmonicTrap:
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
        grid: dvr.DVRSpecification = dvr.add_harmdvr(400, 0.0, 1.0),
    ):
        self.parameters = parameters
        self.grid = grid

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("g", 0.1, "strength of the contact interaction between the particles"),
                ("omega", 1.0, "angular frequency of the harmonic trap"),
            ]
        )

    def get_kinetic_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid,),
            {"kinetic_coeff": -0.5},
            {"kinetic": self.grid.get_d2()},
            "kinetic_coeff | 1 kinetic",
        )

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid,),
            {"potential_coeff": 0.5 * (self.parameters.omega ** 2)},
            {"potential": self.grid.get_x() ** 2},
            "potential_coeff | 1 potential",
        )

    def get_hamiltonian_1b(self) -> tasks.OperatorSpecification:
        return self.get_kinetic_operator_1b() + self.get_potential_operator_1b()

    def get_kinetic_operator(self) -> tasks.MBOperatorSpecification:
        if self.grid.is_fft():
            term = {"value": self.grid.get_d2(), "fft": True}
        else:
            term = self.grid.get_d2()
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"kinetic_coeff": -0.5},
            {"kinetic": term},
            "kinetic_coeff | 1 kinetic",
        )

    def get_potential_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"potential_coeff": 0.5 * (self.parameters.omega ** 2)},
            {"potential": self.grid.get_x() ** 2},
            "potential_coeff | 1 potential",
        )

    def get_interaction_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"interaction_coeff": self.parameters.g},
            {"interaction": self.grid.get_delta()},
            "interaction_coeff | {1:1} interaction",
        )

    def get_hamiltonian(self) -> tasks.MBOperatorSpecification:
        if self.parameters.g != 0.0:
            return (
                self.get_kinetic_operator()
                + self.get_potential_operator()
                + self.get_interaction_operator()
            )

        return self.get_kinetic_operator() + self.get_potential_operator_1b()
