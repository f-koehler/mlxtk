from __future__ import annotations

from numpy.typing import ArrayLike
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


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
