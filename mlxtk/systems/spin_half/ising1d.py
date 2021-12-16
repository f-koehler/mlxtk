from __future__ import annotations

from numpy.typing import ArrayLike
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification
from mlxtk import dvr


class Ising1D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".Ising1D")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("L", 4, "number of sites"),
                ("pbc", True, "whether to use periodic boundary conditions"),
                ("J", 1.0, "ising coupling constant"),
                ("hx", 1.0, "transversal field in x direction"),
                ("hy", 0.0, "transversal field in y direction"),
                ("hz", 0.0, "longitudinal field in z direction"),
            ],
        )

    def create_hamiltonian(self) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        if self.parameters["J"] != 0.0:
            coeffs.update({"-J": -self.parameters["J"]})
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for i in range(
                self.parameters.L if self.parameters["pbc"] else self.parameters.L - 1,
            ):
                j = (i + 1) % self.parameters.L
                table.append(f"-J | {i+1} sz | {j+1} sz")

        if self.parameters["hx"] != 0.0:
            coeffs.update({"-hx": -self.parameters["hx"]})
            terms.update({"sx": self.grid.get().get_sigma_x()})
            for i in range(self.parameters.L):
                table.append(f"-hx | {i+1} sx")

        if self.parameters["hy"] != 0.0:
            coeffs.update({"-hy": -self.parameters["hy"]})
            terms.update({"sy": self.grid.get().get_sigma_y()})
            for i in range(self.parameters.L):
                table.append(f"-hy | {i+1} sy")

        if self.parameters["hz"] != 0.0:
            coeffs.update({"-hz": -self.parameters["hz"]})
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for i in range(self.parameters.L):
                table.append(f"-hz | {i+1} sz")

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            coeffs,
            terms,
            table,
        )

    def create_one_point_operator_x(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"one_point_coeff_x_{site}": 1.0},
            {f"one_point_term_x_{site}": self.grid.get().get_sigma_x()},
            f"one_point_coeff_x_{site} | {site + 1} one_point_term_x_{site}",
        )

    def create_one_point_operator_z(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"one_point_coeff_z_{site}": 1.0},
            {f"one_point_term_z_{site}": self.grid.get().get_sigma_z()},
            f"one_point_coeff_z_{site} | {site + 1} one_point_term_z_{site}",
        )
