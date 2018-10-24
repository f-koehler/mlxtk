from typing import Union

import numpy

from mlxtk import dvr, tasks
from mlxtk.parameters import Parameters

from .single_species import SingleSpeciesSystem


def gaussian(x: Union[float, numpy.ndarray], x0: float):
    return numpy.exp(-((x - x0) ** 2))


class GaussianTrap:
    @staticmethod
    def create_parameters():
        return Parameters(
            [
                ("V0", 1.0, "depth of the Gaussian well"),
                ("x0", 0.0, "center of the Gaussian well"),
                ("g", 0.1, "strength of the contact interaction"),
            ]
        )

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid,),
            {"potential_coeff": -self.parameters.V0},
            {"potential": gaussian(self.grid.get_x(), self.grid.get_x())},
            ["potential_coeff | 1 potential"],
        )

    def get_1b_hamiltonian(self) -> tasks.OperatorSpecification:
        return self.get_kinetic_operator_1b() + self.get_potential_operator_1b()

    def get_potential_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"potential_coeff": -self.parameters.V0},
            {"potential": gaussian(self.grid.get_x(), self.parameters.x0)},
            "potential_coeff | 1 potential",
        )

    def get_interaction_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"interaction_coeff": self.parameters.g},
            {"interaction": self.grid.get_delta()},
            "interaction_coeff | {1:1} interaction",
        )

    def get_hamiltonian(self) -> tasks.MBOperatorSpecification:
        if self.parameters.g != 0.0:
            return (
                self.get_kinetic_operator()
                + self.get_potential_operator()
                + self.get_interaction_operator()
            )

        return self.get_kinetic_operator() + self.get_potential_operator_1b()
