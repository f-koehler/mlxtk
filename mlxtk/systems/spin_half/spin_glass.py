from numpy.typing import ArrayLike
import numpy
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk.parameters import Parameters
from mlxtk.log import get_logger
from mlxtk.tasks import OperatorSpecification
from mlxtk import dvr

class DisorderedXYSpinGlass:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".DisorderedXYSpinGlass")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("L", 4, "number of sites"),
                ("Jmin", -1.0, "minimal coupling constant"),
                ("Jmax", 1.0, "maximal coupling constant"),
                ("alpha", 2.0, "exponent of the interaction term"),
                ("seed", 0, "seed for the disorder")
            ],
        )

    def create_hamiltonian(self) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        dvr: SpinHalfDvr = self.grid.get()

        L = self.parameters["L"]
        prng = numpy.random.Generator(numpy.random.PCG64(self.parameters["seed"]))

        Jmatrix = numpy.zeros((L, L), dtype=numpy.float64)
        dist_matrix = numpy.ones((L, L), dtype=numpy.float64)
        for i in range(L):
            for j in range(i):
                Jmatrix[i,j] = prng.uniform(self.parameters["Jmin"], self.parameters["Jmax"])
                Jmatrix[j,i] = Jmatrix[i,j]
                dist_matrix[i,j] = numpy.abs(i-j)
        coupling_matrix = Jmatrix / (dist_matrix ** self.parameters["alpha"])

        terms.update({"s+": dvr.get_sigma_plus(), "s-": dvr.get_sigma_minus()})
        for i in range(L):
            for j in range(i):
                coeffs.update({f"J_{i}_{j}": coupling_matrix[i,j]})
                table.append(f"J_{i}_{j} | {i+1} s+ | {j+1} s-")
                table.append(f"J_{i}_{j} | {i+1} s- | {j+1} s+")

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            coeffs,
            terms,
            table,
        )


