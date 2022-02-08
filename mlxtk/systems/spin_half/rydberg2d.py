from __future__ import annotations

import numpy
from numpy.typing import ArrayLike
from QDTK.Spin.Primitive import SpinHalfDvr

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class Rydberg2D:
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".Rydberg2D")
        self.parameters = parameters
        self.grid = dvr.add_spin_half_dvr()

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("Lx", 4, "width of the lattice"),
                ("Ly", 4, "height of the lattice"),
                ("delta", 1.0, "laser detuning"),
                ("Rb", 1.0, "blockade radius"),
                ("Rc", 4.0, "cutoff for long-range interactions"),
            ],
        )

    def create_hamiltonian(
        self,
        dof_map: dict[tuple[int, int], int],
    ):
        table: list[str] = []
        coeffs: dict[str, complex] = {}
        terms: dict[str, ArrayLike] = {}

        Lx = self.parameters["Lx"]
        Ly = self.parameters["Ly"]
        Rb = self.parameters["Rb"]
        Rc = self.parameters["Rc"]
        delta = self.parameters["delta"]

        if Rb != 0.0:
            terms.update({"pup": self.grid.get().get_projector_up()})
            for y1 in range(Ly):
                for x1 in range(Lx):
                    i1 = dof_map[(x1, y1)]
                    for y2 in range(Ly):
                        for x2 in range(Lx):
                            i2 = dof_map[(x2, y2)]
                            if i2 >= i1:
                                continue

                            dx = x2 - x1
                            dy = y2 - y1

                            r = numpy.sqrt((dx**2) + (dy**2))
                            if (Rc > 0.0) and (r >= Rc):
                                continue

                            prefactor = 0.5 * ((Rb / r) ** 6)
                            coeffs[f"prefactor_{i1}_{i2}"] = prefactor
                            table.append(
                                f"prefactor_{i1}_{i2} | {i1+1} pup | {i2+1} pup",
                            )

        coeffs.update({"0.5": 0.5})
        terms.update({"sx": self.grid.get().get_sigma_x()})
        for i in range(Lx * Ly):
            table.append(f"0.5 | {i+1} sx")

        if delta != 0.0:
            coeffs.update({"-delta": -delta})
            terms.update({"pup": self.grid.get().get_projector_up()})
            for i in range(Lx * Ly):
                table.append(f"-delta | {i+1} pup")

        return OperatorSpecification([self.grid] * (Lx * Ly), coeffs, terms, table)
