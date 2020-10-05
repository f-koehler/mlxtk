from abc import ABC, abstractmethod
from typing import List

from QDTK.SQR.Primitive import SQRDvrBosonic

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import OperatorSpecification


class BosonicSQR(ABC):
    def __init__(self, parameters: Parameters):
        self.logger = get_logger(__name__ + ".BosonicSQR")
        self.parameters = parameters
        self.grid = dvr.add_sqr_dvr_bosonic(parameters.N)

    def get_particle_number_operator(self) -> OperatorSpecification:

        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {"particle_number_coeff": 1.0},
            {"particle_number": self.grid.get_x()},
            [
                "particle_number_coeff | {} particle_number".format(i + 1)
                for i in range(self.parameters.sites)
            ],
        )

    def get_penalty_term(self, penalty: float) -> OperatorSpecification:
        table: List[str] = []

        for i in range(self.parameters.sites):
            table.append("penalty_coeff_1 | {} penalty_1".format(i + 1))
            for j in range(i):
                table.append(
                    "penalty_coeff_2 | {} penalty_2 | {} penalty_2".format(i + 1, j + 1)
                )
            table.append("penalty_coeff_3 | {} penalty_3".format(i + 1))

        table.append("penalty_coeff_4 | 1 penalty_4")

        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {
                "penalty_coeff_1": penalty,
                "penalty_coeff_2": 2 * penalty,
                "penalty_coeff_3": -2 * penalty * self.parameters.N,
                "penalty_coeff_4": penalty * (self.parameters.N ** 2),
            },
            {
                "penalty_1": self.grid.get_x() ** 2,
                "penalty_2": self.grid.get_x(),
                "penalty_3": self.grid.get_x(),
                "penalty_4": self.grid.get().get_unit_operator(),
            },
            table,
        )

    @staticmethod
    @abstractmethod
    def create_parameters() -> Parameters:
        pass
