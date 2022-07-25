from __future__ import annotations

import numpy
from numpy.typing import ArrayLike

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class IsingLR1D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".IsingLR1D")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("L", 4, "number of sites"),
                ("J", 1.0, "interaction strength"),
                ("hx", 1.0, "transversal field in x direction"),
                ("hy", 0.0, "transversal field in y direction"),
                ("hz", 0.0, "longitudinal field in z direction"),
                ("alpha", 6.0, "exponent for the interaction"),
                (
                    "cutoff_radius",
                    -1.0,
                    "cutoff_radius for the interaction (negative number => no cutoff)",
                ),
            ],
        )

    def create_hamiltonian(self) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        if self.parameters["J"] != 0.0:
            terms.update({"sz": self.grid.get().get_sigma_z()})
            for i in range(
                self.parameters.L,
            ):
                for j in range(i):
                    if (self.parameters["cutoff_radius"] > 0.0) and (
                        numpy.abs(i - j) > self.parameters["cutoff_radius"]
                    ):
                        continue

                    coupling = -self.parameters["J"] / (
                        numpy.abs(i - j) ** self.parameters["alpha"]
                    )

                    coeffs.update({f"-J_{i}_{j}": coupling})
                    table.append(f"-J_{i}_{j} | {i+1} sz | {j+1} sz")

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

    def create_all_up_projector(self) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"all_up_coeff": 1.0},
            {f"all_up_term": self.grid.get().get_projector_up()},
            "all_up_coeff "
            + " ".join(f"| {site+1} all_up_term" for site in range(self.parameters.L)),
        )

    def create_all_down_projector(self) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"all_down_coeff": 1.0},
            {f"all_down_term": self.grid.get().get_projector_down()},
            "all_down_coeff "
            + " ".join(
                f"| {site+1} all_down_term" for site in range(self.parameters.L)
            ),
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
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * self.parameters.L,
                {f"sx_sx_coeff_{site1}_{site2}": 1.0},
                {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sx_sx_coeff_{site1}_{site2} | {site1 + 1} sx_sx_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sx_sx_coeff_{site1}_{site2}": 1.0},
            {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_x()},
            f"sx_sx_coeff_{site1}_{site2} | {site1 + 1} sx_sx_term_{site1}_{site2}| {site2 + 1} sx_sx_term_{site1}_{site2}",
        )

    def create_sy_sy_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * self.parameters.L,
                {f"sy_sy_coeff_{site1}_{site2}": 1.0},
                {f"sy_sy_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sy_sy_coeff_{site1}_{site2} | {site1 + 1} sy_sy_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sy_sy_coeff_{site1}_{site2}": 1.0},
            {f"sy_sy_term_{site1}_{site2}": self.grid.get().get_sigma_y()},
            f"sy_sy_coeff_{site1}_{site2} | {site1 + 1} sy_sy_term_{site1}_{site2}| {site2 + 1} sy_sy_term_{site1}_{site2}",
        )

    def create_sz_sz_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * self.parameters.L,
                {f"sz_sz_coeff_{site1}_{site2}": 1.0},
                {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sz_sz_coeff_{site1}_{site2} | {site1 + 1} sz_sz_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sz_sz_coeff_{site1}_{site2}": 1.0},
            {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_z()},
            f"sz_sz_coeff_{site1}_{site2} | {site1 + 1} sz_sz_term_{site1}_{site2}| {site2 + 1} sz_sz_term_{site1}_{site2}",
        )
