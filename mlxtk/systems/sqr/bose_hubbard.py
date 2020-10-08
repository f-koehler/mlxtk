from typing import List, Optional

from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.systems.sqr.bosonic import BosonicSQR
from mlxtk.tasks import OperatorSpecification


class BoseHubbardSQR(BosonicSQR):
    def __init__(self, parameters: Parameters):
        super().__init__(parameters)
        self.logger = get_logger(__name__ + ".BoseHubbardSQR")

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("sites", 4, "number of sites"),
                ("N", 4, "number of particles"),
                ("J", 1.0, "hopping constant"),
                ("U", 1.0, "interaction strength"),
                ("pbc", True, "whether to use periodic boundary conditions"),
            ]
        )

    def create_hopping_term(self) -> OperatorSpecification:
        table: List[str] = []

        for i in range(self.parameters.sites - 1):
            table += [
                "hopping_coeff | {} creator | {} annihilator".format(i + 1, i + 2),
                "hopping_coeff | {} creator | {} annihilator".format(i + 2, i + 1),
            ]

        if self.parameters.pbc:
            table += [
                "hopping_coeff | {} creator | {} annihilator".format(
                    self.parameters.sites, 1
                ),
                "hopping_coeff | {} creator | {} annihilator".format(
                    1, self.parameters.sites
                ),
            ]

        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {
                "hopping_coeff": -parameters.J,
            },
            {
                "creator": self.grid.get().get_creation_operator(),
                "annihilator": self.grid.get().get_annihilation_operator(),
            },
            table,
        )

    def create_interaction_term(self) -> OperatorSpecification:
        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {
                "interaction_coeff": self.parameters.U / 2.0,
            },
            {
                "interaction": self.grid.get_x() * (self.grid.get_x() - 1.0),
            },
            [
                "interaction_coeff | {} interaction".format(i + 1)
                for i in range(self.parameters.sites)
            ],
        )

    def create_hamiltonian(
        self, penalty: Optional[float] = None
    ) -> OperatorSpecification:
        hamiltonian = self.get_penalty_term(
            self.parameters.penalty if penalty is None else penalty
        )
        if self.parameters.J != 0.0:
            hamiltonian += self.create_hopping_term()
        if self.parameters.U != 0.0:
            hamiltonian += self.create_interaction_term()
        return hamiltonian
