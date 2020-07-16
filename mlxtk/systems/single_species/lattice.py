from typing import Union

import numpy

from mlxtk import tasks
from mlxtk.parameters import Parameters
from mlxtk.systems.single_species.single_species import SingleSpeciesSystem


class Lattice(SingleSpeciesSystem):
    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("N", 2, "number of particles"),
                ("m", 5, "number of single particle functions"),
                ("g", 0.1, "strength of the contact interaction between the particles"),
                ("wells", 5, "number of wells"),
                ("V0", 1.0, "depth of the lattice"),
            ]
        )

    def get_potential(
        self, x: Union[float, numpy.ndarray]
    ) -> Union[float, numpy.ndarray]:
        if self.parameters.wells % 2 == 0:
            return numpy.cos(numpy.pi * self.parameters.wells * x) ** 2
        return numpy.sin(numpy.pi * self.parameters.wells * x) ** 2

    @staticmethod
    def convert_recoil_energy(
        energy: Union[float, numpy.ndarray], parameters: Parameters
    ) -> Union[float, numpy.ndarray]:
        return 0.5 * (numpy.pi ** 2) * (parameters.wells ** 2) * energy

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid_1b,),
            {"potential_coeff": self.parameters.V0},
            {"potential": self.get_potential(self.grid_1b.get_x())},
            "potential_coeff | 1 potential",
        )

    def get_hamiltonian_1b(self) -> tasks.OperatorSpecification:
        return self.get_kinetic_operator_1b() + self.get_potential_operator_1b()

    def get_potential_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"potential_coeff": self.parameters.V0},
            {"potential": self.get_potential(self.grid_1b.get_x())},
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
        if self.parameters.g != 0.0 and self.parameters.N > 1:
            return (
                self.get_kinetic_operator()
                + self.get_potential_operator()
                + self.get_interaction_operator()
            )

        return self.get_kinetic_operator() + self.get_potential_operator()
