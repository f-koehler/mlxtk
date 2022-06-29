from __future__ import annotations

from typing import Mapping
from numpy.typing import ArrayLike
import numpy

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class Vitagliano2010:
    def __init__(self, parameters: Parameters, dof_map: Mapping[int,int]):
        self.logger = get_logger(__name__ + ".Vitagliano2010")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()
        self.dof_map = dof_map

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("L", 4, "number of sites"),
                ("J0", 1.0, "central coupling constant"),
                ("function", "gaussian", "function for the coupling constants [uniform, exponential, gaussian]"),
            ],
        )

    def create_hamiltonian(self) -> OperatorSpecification:
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        L = self.parameters["L"]
        J0 = self.parameters["J0"]
        funcname = self.parameters["function"]

        terms["sx"] = self.grid.get().get_sigma_x()
        terms["sy"] = self.grid.get().get_sigma_y()

        if funcname == "uniform":
            function = lambda n: 1.0
        elif funcname == "exponential":
            function = lambda n: numpy.exp(-n)
        elif funcname == "gaussian":
            function = lambda n: numpy.exp(-2*n*n)
        else:
            raise ValueError("Invalid function name")

        for i in range(L-1):
            dof_1 = self.dof_map[i]
            dof_2 = self.dof_map[i+1]
            n = abs(L//2-1-i)
            coeffs[f"J_{i}"] = 0.5 * J0 * function(n)
            table.append(f"J_{i} | {dof_1+1} sx | {dof_2+1} sx")
            table.append(f"J_{i} | {dof_1+1} sy | {dof_2+1} sy")

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            coeffs,
            terms,
            table,
        )

    def create_sx_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sx_coeff_{site}": 1.0},
            {f"sx_term_{site}": self.grid.get().get_sigma_x()},
            f"sx_coeff_{site} | {self.dof_map[site] + 1} sx_term_{site}",
        )

    def create_sy_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sy_coeff_{site}": 1.0},
            {f"sy_term_{site}": self.grid.get().get_sigma_y()},
            f"sy_coeff_{site} | {self.dof_map[site] + 1} sy_term_{site}",
        )

    def create_sz_operator(self, site: int) -> OperatorSpecification:
        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sz_coeff_{site}": 1.0},
            {f"sz_term_{site}": self.grid.get().get_sigma_z()},
            f"sz_coeff_{site} | {self.dof_map[site] + 1} sz_term_{site}",
        )

    def create_sx_sx_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * self.parameters.L,
                {f"sx_sx_coeff_{site1}_{site2}": 1.0},
                {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sx_sx_coeff_{site1}_{site2} | {self.dof_map[site1] + 1} sx_sx_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sx_sx_coeff_{site1}_{site2}": 1.0},
            {f"sx_sx_term_{site1}_{site2}": self.grid.get().get_sigma_x()},
            f"sx_sx_coeff_{site1}_{site2} | {self.dof_map[site1] + 1} sx_sx_term_{site1}_{site2}| {self.dof_map[site2] + 1} sx_sx_term_{site1}_{site2}",
        )

    def create_sy_sy_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * self.parameters.L,
                {f"sy_sy_coeff_{site1}_{site2}": 1.0},
                {f"sy_sy_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sy_sy_coeff_{site1}_{site2} | {self.dof_map[site1] + 1} sy_sy_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sy_sy_coeff_{site1}_{site2}": 1.0},
            {f"sy_sy_term_{site1}_{site2}": self.grid.get().get_sigma_y()},
            f"sy_sy_coeff_{site1}_{site2} | {self.dof_map[site1] + 1} sy_sy_term_{site1}_{site2}| {self.dof_map[site2] + 1} sy_sy_term_{site1}_{site2}",
        )

    def create_sz_sz_operator(self, site1: int, site2: int) -> OperatorSpecification:
        if site1 == site2:
            return OperatorSpecification(
                [self.grid] * self.parameters.L,
                {f"sz_sz_coeff_{site1}_{site2}": 1.0},
                {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_0()},
                f"sz_sz_coeff_{site1}_{site2} | {self.dof_map[site1] + 1} sz_sz_term_{site1}_{site2}",
            )

        return OperatorSpecification(
            [self.grid] * self.parameters.L,
            {f"sz_sz_coeff_{site1}_{site2}": 1.0},
            {f"sz_sz_term_{site1}_{site2}": self.grid.get().get_sigma_z()},
            f"sz_sz_coeff_{site1}_{site2} | {self.dof_map[site1] + 1} sz_sz_term_{site1}_{site2}| {self.dof_map[site2] + 1} sz_sz_term_{site1}_{site2}",
        )
