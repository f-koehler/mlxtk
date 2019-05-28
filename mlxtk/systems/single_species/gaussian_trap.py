from typing import Union

import numpy

from mlxtk import tasks
from mlxtk.parameters import Parameters

from .single_species import SingleSpeciesSystem


def gaussian(x: Union[float, numpy.ndarray], x0: float, alpha: float = 1.0):
    return numpy.exp(-((x - x0)**2) / (alpha**2))


class GaussianTrap(SingleSpeciesSystem):
    @staticmethod
    def create_parameters():
        return Parameters([
            ("N", 2, "number of particles"),
            ("m", 5, "number of single particle functions"),
            ("V0", 1.0, "depth of the Gaussian well"),
            ("x0", 0.0, "center of the Gaussian well"),
            ("g", 0.1, "strength of the contact interaction"),
        ])

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return self.create_gaussian_potential_operator_1b(
            self.parameters.x0, self.parameters.V0)

    def get_hamiltonian_1b(self) -> tasks.OperatorSpecification:
        return self.get_kinetic_operator_1b() + self.get_potential_operator_1b(
        )

    def get_potential_operator(self) -> tasks.MBOperatorSpecification:
        return self.create_gaussian_potential_operator(self.parameters.x0,
                                                       self.parameters.V0)

    def get_interaction_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1, ),
            (self.grid, ),
            {"interaction_coeff": self.parameters.g},
            {"interaction": self.grid.get_delta()},
            "interaction_coeff | {1:1} interaction",
        )

    def get_hamiltonian(self) -> tasks.MBOperatorSpecification:
        if self.parameters.g != 0.0:
            return (self.get_kinetic_operator() +
                    self.get_potential_operator() +
                    self.get_interaction_operator())

        return self.get_kinetic_operator() + self.get_potential_operator()

    def create_gaussian_potential_operator_1b(self,
                                              x0: float,
                                              V0: float,
                                              name: str = "potential",
                                              alpha: float = 1.0
                                              ) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid_1b, ),
            {name + "_coeff": -V0},
            {name: gaussian(self.grid.get_x(), x0, alpha)},
            name + "_coeff | 1 " + name,
        )

    def create_gaussian_potential_operator(self,
                                           x0: float,
                                           V0: float,
                                           name: str = "potential",
                                           alpha: float = 1.0
                                           ) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1, ),
            (self.grid, ),
            {name + "_coeff": -V0},
            {name: gaussian(self.grid.get_x(), x0, alpha)},
            name + "_coeff | 1 " + name,
        )

    def get_hamiltonian_moving_well(self, v: float = 1.0, a: float = 0.0
                                    ) -> tasks.MBOperatorSpecification:
        potential = tasks.MBOperatorSpecification(
            (1, ), (self.grid, ), {"potential_coeff": 1.0}, {
                "potential": {
                    "td_name": "moving_gaussian",
                    "td_args": [-self.parameters.V0, self.parameters.x0, v, a]
                }
            }, "potential_coeff | 1 potential")
        return self.get_kinetic_operator(
        ) + potential + self.get_interaction_operator()
