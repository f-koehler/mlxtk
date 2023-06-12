from numpy.typing import ArrayLike

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class FermiHubbard1D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".FermiHubbard1D")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("L", 4, "number of sites"),
                ("J", 1.0, "hopping amplitude"),
                ("U", 1.0, "interaction strength"),
            ],
        )

    def create_hamiltonian(self) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        terms.update(
            {
                "sx": self.grid.get().get_sigma_x(),
                "sy": self.grid.get().get_sigma_y(),
                "szp1": self.grid.get().get_sigma_z() + self.grid.get().get_identity(),
            },
        )

        if self.parameters["J"] != 0.0:
            coeffs.update({"-J": -self.parameters["J"] / 2.0})
            for i in range(self.parameters.L - 1):
                table.append(f"-J | {2*i+1} sx | {2*i+3} sx")
                table.append(f"-J | {2*i+1} sy | {2*i+3} sy")
                table.append(f"-J | {2*i+2} sx | {2*i+4} sx")
                table.append(f"-J | {2*i+2} sy | {2*i+4} sy")

        if self.parameters["U"] != 0.0:
            coeffs.update({"U": self.parameters["U"] / 4.0})
            for i in range(self.parameters.L):
                table.append(f"U | {2*i+1} szp1 | {2*i+2} szp1")

        return OperatorSpecification(
            [self.grid] * (2 * self.parameters.L),
            coeffs,
            terms,
            table,
        )
