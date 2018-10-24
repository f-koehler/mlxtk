from mlxtk import dvr, tasks
from mlxtk.parameters import Parameters

from .single_species import SingleSpeciesSystem


class HarmonicTrap(SingleSpeciesSystem):
    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("N", 2, "number of particles"),
                ("m", 5, "number of single particle functions"),
                ("g", 0.1, "strength of the contact interaction between the particles"),
                ("omega", 1.0, "angular frequency of the harmonic trap"),
            ]
        )

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return tasks.OperatorSpecification(
            (self.grid_1b,),
            {"potential_coeff": 0.5 * (self.parameters.omega ** 2)},
            {"potential": self.grid_1b.get_x() ** 2},
            "potential_coeff | 1 potential",
        )

    def get_hamiltonian_1b(self) -> tasks.OperatorSpecification:
        return self.get_kinetic_operator_1b() + self.get_potential_operator_1b()

    def get_potential_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"potential_coeff": 0.5 * (self.parameters.omega ** 2)},
            {"potential": self.grid.get_x() ** 2},
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

        return self.get_kinetic_operator() + self.get_potential_operator()

    def get_center_of_mass_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"center_of_mass_coeff": 1.0},
            {"center_of_mass": self.grid.get_x()},
            "center_of_mass_coeff | 1 center_of_mass",
        )

    def get_center_of_mass_operator_squared(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1,),
            (self.grid,),
            {
                "center_of_mass_squared_coeff_one": 1.0,
                "center_of_mass_squared_coeff_two": 2.0,
            },
            {
                "center_of_mass_squared_x": self.grid.get_x(),
                "center_of_mass_squared_x^2": self.grid.get_x() ** 2,
            },
            [
                "center_of_mass_squared_coeff_two | 1 center_of_mass_squared_x",
                "center_of_mass_squared_coeff_one | 1 center_of_mass_squared_x^2",
            ],
        )
