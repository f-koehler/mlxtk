from decimal import DivisionByZero
from typing import List, Optional

import numpy

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
            ],
        )

    def create_hopping_term(self) -> OperatorSpecification:
        table: List[str] = []

        for i in range(self.parameters.sites - 1):
            table += [
                f"hopping_coeff | {i + 1} creator | {i + 2} annihilator",
                f"hopping_coeff | {i + 2} creator | {i + 1} annihilator",
            ]

        if self.parameters.pbc:
            table += [
                "hopping_coeff | {} creator | {} annihilator".format(
                    self.parameters.sites,
                    1,
                ),
                "hopping_coeff | {} creator | {} annihilator".format(
                    1,
                    self.parameters.sites,
                ),
            ]

        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {
                "hopping_coeff": -self.parameters.J,
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
                f"interaction_coeff | {i + 1} interaction"
                for i in range(self.parameters.sites)
            ],
        )

    def create_hamiltonian(
        self,
        penalty: Optional[float] = None,
    ) -> OperatorSpecification:
        terms: List[OperatorSpecification] = []
        penalty = self.parameters.penalty if penalty is None else penalty
        if penalty != 0.0:
            terms.append(self.get_penalty_term(penalty))
        if self.parameters.J != 0.0:
            terms.append(self.create_hopping_term())
        if self.parameters.U != 0.0:
            terms.append(self.create_interaction_term())
        if not terms:
            raise RuntimeError("Hamiltonian would be empty: U, J and penalty are 0.0")
        hamiltonian = terms[0]
        for term in terms[1:]:
            hamiltonian += term
        return hamiltonian

    def create_efficient_hamiltonian(
        self,
        gamma: Optional[float] = None,
        zeta: Optional[float] = None,
    ) -> OperatorSpecification:
        terms: List[OperatorSpecification] = []
        gamma = self.parameters.gamma if gamma is None else gamma
        zeta = self.parameters.zeta if zeta is None else zeta

        if gamma != 0.0:
            if zeta == 0.0:
                raise DivisionByZero("zeta must be positive")
            terms.append(self.get_efficient_penalty_term(gamma, zeta))

        if self.parameters.J != 0.0:
            terms.append(self.create_hopping_term())
        if self.parameters.U != 0.0:
            terms.append(self.create_interaction_term())
        if not terms:
            raise RuntimeError("Hamiltonian would be empty: U, J and gamma are 0.0")

        hamiltonian = terms[0]
        for term in terms[1:]:
            hamiltonian += term
        return hamiltonian

    def get_initial_filling(self) -> numpy.ndarray:
        N = self.parameters["N"]
        L = self.parameters["sites"]

        if N % L == 0:
            return numpy.full(L, N // L, self, dtype=numpy.int64)

        fillings = numpy.zeros(self.parameters.sites, dtype=numpy.int64)
        loc = 0
        for _ in range(self.parameters.N):
            fillings[loc] += 1
            loc = (loc + 1) % self.parameters.sites
        return numpy.roll(fillings, (L - loc) // 2)
