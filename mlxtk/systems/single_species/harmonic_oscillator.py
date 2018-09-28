from mlxtk import dvr, tasks


class HarmonicOscillator(object):
    """A class for a quantum harmonic oscillator with interacting particles

    Args:
        mass (float): mass of the particles
        omega (float): angular frequency of the harmonic trap
        g (float): strength of the inter-particle contact interaction
        grid (mlxtk.dvr.DVRSpecification): the grid of the harmonic oscillator

    Attributes:
        mass (float): mass of the particles
        omega (float): angular frequency of the harmonic trap
        g (float): strength of the inter-particle contact interaction
        grid (mlxtk.dvr.DVRSpecification): the grid of the harmonic oscillator
    """

    def __init__(self, mass=1., omega=1., g=0.1, grid=dvr.add_harmdvr(400, 0., 1.)):
        self.mass = mass
        self.omega = omega
        self.grid = grid
        self.g = g

    def get_1b_hamiltonian(self, name):
        """Create the single-body Hamiltonian
        Args:
           name (str): name for the operator file

        Returns:
           list: list of task dictionaries to create this operator
        """
        return tasks.create_operator(
            name,
            (self.grid,),
            {"kin": -1. / (2. * self.mass), "pot": 0.5 * self.mass * (self.omega ** 2)},
            {"dx²": self.grid.get_d2(), "x²": self.grid.get_x() ** 2},
            ["kin | 1 dx²", "pot | 1 x²"],
        )

    def get_mb_hamiltonian(self, name, **kwargs):
        """Create the many body hamiltonian

        The system parameters can be changed for this operator.
        This is useful when quenching a system parameter.

        Args:
           name (str): name for the operator file
           mass (float): mass of the particles
           omega (float): angular frequency of the harmonic trap
           g (float): strength of the inter-particle contact interaction

        Returns:
           list: list of task dictionaries to create this operator
        """
        mass = kwargs.get("mass", self.mass)
        omega = kwargs.get("omega", self.omega)
        g = kwargs.get("g", self.g)

        coefficients = {"kin": 1. / (2. * mass), "pot": 0.5 * mass * (omega ** 2)}
        terms = {"dx²": self.grid.get_d2(), "x²": self.grid.get_x() ** 2}
        table = ["kin | 1 dx²", "pot | 1 x²"]

        if g != 0.0:
            coefficients["int"] = g
            terms["delta"] = self.grid.get_delta()
            table.append("int | {1:1} delta")

        return tasks.create_many_body_operator(
            name, (1,), (self.grid,), coefficients, terms, table
        )
