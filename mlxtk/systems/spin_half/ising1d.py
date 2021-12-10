from __future__ import annotations

from numpy.typing import ArrayLike
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class Ising1D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".Ising1D")
        self.parameters = parameters
        self.grid = SpinHalfDvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("sites", 4, "number of sites"),
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
            terms.update({"sz": self.grid.get_sigma_z()})
            for i in range(
                self.parameters.sites
                if self.parameters["pbc"]
                else self.parameters.sites - 1,
            ):
                table += f"-J | {i} sz | {(i+1) % self.parameters.sites} sz"

        if self.parameters["hx"] != 0.0:
            coeffs.update({"-hx": -self.parameters["hx"]})
            terms.update({"sx": self.grid.get_sigma_x()})
            for i in range(self.parameters.sites):
                table += f"-hx | {i} sx"

        if self.parameters["hy"] != 0.0:
            coeffs.update({"-hy": -self.parameters["hy"]})
            terms.update({"sy": self.grid.get_sigma_y()})
            for i in range(self.parameters.sites):
                table += f"-hy | {i} sy"

        if self.parameters["hz"] != 0.0:
            coeffs.update({"-hz": -self.parameters["hz"]})
            terms.update({"sz": self.grid.get_sigma_z()})
            for i in range(self.parameters.sites):
                table += f"-hz | {i} sz"

        return OperatorSpecification(
            [self.grid] * self.parameters.sites,
            coeffs,
            terms,
            table,
        )
