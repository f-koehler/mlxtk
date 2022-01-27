from __future__ import annotations

from numpy.typing import ArrayLike
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class Ising2D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".Ising2D")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("L", 4, "width and height of the lattice"),
                ("pbc", True, "whether to use periodic boundary conditions"),
                ("Jx", 1.0, "Ising coupling constant in x direction"),
                ("Jy", 1.0, "Ising coupling constant in y direction"),
                ("hx", 1.0, "transversal field in x direction"),
                ("hy", 0.0, "transversal field in y direction"),
                ("hz", 0.0, "longitudinal field in z direction"),
            ],
        )

    def create_hamiltonian(
        self,
        dof_map: dict[tuple[int, int], int],
    ) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        L = self.parameters["L"]

        if self.parameters["Jx"] != 0.0:
            coeffs.update({"-Jx": -self.parameters["Jx"]})
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for y in range(L):
                for x in range(L if self.parameters["pbc"] else L - 1):
                    index1 = dof_map[(x, y)]
                    index2 = dof_map[((x + 1) % L, y)]
                    table.append(f"-Jx | {index1 + 1} sz | {index2 + 1} sz")

        if self.parameters["Jy"] != 0.0:
            coeffs.update({"-Jy": -self.parameters["Jy"]})
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for y in range(L if self.parameters["pbc"] else L - 1):
                for x in range(L):
                    index1 = dof_map[(x, y)]
                    index2 = dof_map[(x, (y + 1) % L)]
                    table.append(f"-Jy | {index1 + 1} sz | {index2 + 1} sz")

        if self.parameters["hx"] != 0.0:
            coeffs.update({"-hx": -self.parameters["hx"]})
            terms.update({"sx": self.grid.get().get_sigma_x()})
            for i in range(L ** 2):
                table.append(f"-hx | {i+1} sx")

        if self.parameters["hy"] != 0.0:
            coeffs.update({"-hy": -self.parameters["hy"]})
            terms.update({"sy": self.grid.get().get_sigma_y()})
            for i in range(L ** 2):
                table.append(f"-hy | {i+1} sy")

        if self.parameters["hz"] != 0.0:
            coeffs.update({"-hz": -self.parameters["hz"]})
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for i in range(L ** 2):
                table.append(f"-hz | {i+1} sz")

        return OperatorSpecification(
            [self.grid] * (self.parameters.L ** 2),
            coeffs,
            terms,
            table,
        )

    def create_Sx_operator(self) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * (self.parameters.L ** 2),
            {f"Sx_coeff": 1.0},
            {f"Sx_term": self.grid.get().get_sigma_x()},
            [
                f"Sx_coeff | {site + 1} Sx_term"
                for site in range(self.parameters["L"] ** 2)
            ],
        )

    def create_Sz_operator(self) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * (self.parameters.L ** 2),
            {f"Sz_coeff": 1.0},
            {f"Sz_term": self.grid.get().get_sigma_z()},
            [
                f"Sz_coeff | {site + 1} Sz_term"
                for site in range(self.parameters["L"] ** 2)
            ],
        )