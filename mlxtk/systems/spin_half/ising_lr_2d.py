from __future__ import annotations

import itertools

import numpy
from numpy.typing import ArrayLike

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class IsingLR2D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".IsingLR2D")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("Lx", 4, "width and height of the lattice"),
                ("Ly", 4, "width and height of the lattice"),
                ("J", 1.0, "interaction strength"),
                ("hx", 1.0, "transversal field in x direction"),
                ("hy", 0.0, "transversal field in y direction"),
                ("hz", 0.0, "longitudinal field in z direction"),
                ("alpha", 6.0, "exponent for the interaction"),
                (
                    "cutoff_radius",
                    -1.0,
                    "cutoff radius for the interaction (negative number => no cutoff)",
                ),
            ],
        )

    def create_hamiltonian(
        self,
        dof_map: dict[tuple[int, int], int],
    ) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        Lx: int = self.parameters["Lx"]
        Ly: int = self.parameters["Ly"]
        rc: float = self.parameters["cutoff_radius"]
        J: float = self.parameters["J"]
        alpha: float = self.parameters["alpha"]
        N = Lx * Ly

        if self.parameters["J"] != 0.0:
            for (x1, y1) in itertools.product(range(Lx), range(Ly)):
                i = dof_map[(x1, y1)]
                for (x2, y2) in itertools.product(range(Lx), range(Ly)):
                    j = dof_map[(x2, y2)]
                    if i <= j:
                        continue

                    dx = x1 - x2
                    dy = y1 - y2
                    dist = numpy.sqrt((dx**2) + (dy**2))
                    if (rc > 0.0) and (dist > rc):
                        continue

                    coupling = -J / (dist**alpha)
                    coeffs.update({f"-J_{i}_{j}": coupling})
                    table.append(f"-J_{i}_{j} | {i+1} sz | {j+1} sz")

        if self.parameters["hx"] != 0.0:
            coeffs.update({"-hx": -self.parameters["hx"]})
            terms.update({"sx": self.grid.get().get_sigma_x()})
            for i in range(N):
                table.append(f"-hx | {i+1} sx")

        if self.parameters["hy"] != 0.0:
            coeffs.update({"-hy": -self.parameters["hy"]})
            terms.update({"sy": self.grid.get().get_sigma_y()})
            for i in range(N):
                table.append(f"-hy | {i+1} sy")

        if self.parameters["hz"] != 0.0:
            coeffs.update({"-hz": -self.parameters["hz"]})
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for i in range(N):
                table.append(f"-hz | {i+1} sz")

        return OperatorSpecification(
            [self.grid] * N,
            coeffs,
            terms,
            table,
        )

    def create_sx_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * (self.parameters.Lx * self.parameters.Ly),
            {f"sx_coeff_{site}": 1.0},
            {f"sx_term_{site}": self.grid.get().get_sigma_x()},
            f"sx_coeff_{site} | {site + 1} sx_term_{site}",
        )

    def create_sy_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * (self.parameters.Lx * self.parameters.Ly),
            {f"sy_coeff_{site}": 1.0},
            {f"sy_term_{site}": self.grid.get().get_sigma_y()},
            f"sy_coeff_{site} | {site + 1} sy_term_{site}",
        )

    def create_sz_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * (self.parameters.Lx * self.parameters.Ly),
            {f"sz_coeff_{site}": 1.0},
            {f"sz_term_{site}": self.grid.get().get_sigma_z()},
            f"sz_coeff_{site} | {site + 1} sz_term_{site}",
        )

    def create_sx_sx_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * (self.parameters.Lx * self.parameters.Ly),
                {f"sx_sx_coeff_{site1}_{site2}": 1.0},
                {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sx_sx_coeff_{site1}_{site2} | {site1 + 1} sx_sx_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * (self.parameters.Lx * self.parameters.Ly),
            {f"sx_sx_coeff_{site1}_{site2}": 1.0},
            {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_x()},
            f"sx_sx_coeff_{site1}_{site2} | {site1 + 1} sx_sx_term_{site1}_{site2}| {site2 + 1} sx_sx_term_{site1}_{site2}",
        )

    def create_sy_sy_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * (self.parameters.Lx * self.parameters.Ly),
                {f"sy_sy_coeff_{site1}_{site2}": 1.0},
                {f"sy_sy_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sy_sy_coeff_{site1}_{site2} | {site1 + 1} sy_sy_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * (self.parameters.Lx * self.parameters.Ly),
            {f"sy_sy_coeff_{site1}_{site2}": 1.0},
            {f"sy_sy_term_{site1}_{site2}": self.grid.get().get_sigma_y()},
            f"sy_sy_coeff_{site1}_{site2} | {site1 + 1} sy_sy_term_{site1}_{site2}| {site2 + 1} sy_sy_term_{site1}_{site2}",
        )

    def create_sz_sz_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * (self.parameters.Lx * self.parameters.Ly),
                {f"sz_sz_coeff_{site1}_{site2}": 1.0},
                {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sz_sz_coeff_{site1}_{site2} | {site1 + 1} sz_sz_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * (self.parameters.Lx * self.parameters.Ly),
            {f"sz_sz_coeff_{site1}_{site2}": 1.0},
            {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_z()},
            f"sz_sz_coeff_{site1}_{site2} | {site1 + 1} sz_sz_term_{site1}_{site2}| {site2 + 1} sz_sz_term_{site1}_{site2}",
        )
