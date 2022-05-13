import numpy
from numpy.typing import ArrayLike
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


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
                ("seed", 0, "seed for the disorder"),
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

    def create_Sx_operator(self) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"Sx_coeff": 1.0},
            {f"Sx_term": self.grid.get().get_sigma_x()},
            [f"Sx_coeff | {site + 1} Sx_term" for site in range(self.parameters["L"])],
        )

    def create_Sz_operator(self) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"Sz_coeff": 1.0},
            {f"Sz_term": self.grid.get().get_sigma_z()},
            [f"Sz_coeff | {site + 1} Sz_term" for site in range(self.parameters["L"])],
        )

    def create_sx_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sx_coeff_{site}": 1.0},
            {f"sx_term_{site}": self.grid.get().get_sigma_x()},
            f"sx_coeff_{site} | {site + 1} sx_term_{site}",
        )

    def create_sy_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sy_coeff_{site}": 1.0},
            {f"sy_term_{site}": self.grid.get().get_sigma_y()},
            f"sy_coeff_{site} | {site + 1} sy_term_{site}",
        )

    def create_sz_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sz_coeff_{site}": 1.0},
            {f"sz_term_{site}": self.grid.get().get_sigma_z()},
            f"sz_coeff_{site} | {site + 1} sz_term_{site}",
        )

    def create_sx_sx_operator(self, site1: int, site2: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sx_sx_coeff_{site1}_{site2}": 1.0},
            {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_x()},
            f"sx_sx_coeff_{site1}_{site2} | {site1 + 1} sx_sx_term_{site1}_{site2}| {site2 + 1} sx_sx_term_{site1}_{site2}",
        )

    def create_sz_sz_operator(self, site1: int, site2: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sz_sz_coeff_{site1}_{site2}": 1.0},
            {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_z()},
            f"sz_sz_coeff_{site1}_{site2} | {site1 + 1} sz_sz_term_{site1}_{site2}| {site2 + 1} sz_sz_term_{site1}_{site2}",
        )
