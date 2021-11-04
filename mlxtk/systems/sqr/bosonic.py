from abc import ABC, abstractmethod
from typing import List

import numpy
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

    def create_correlator(self, site_a: int, site_b: int) -> OperatorSpecification:
        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {"correlator_coeff": 1.0},
            {
                "correlator_creator": self.grid.get().get_creation_operator(),
                "correlator_annihilator": self.grid.get().get_annihilation_operator(),
            },
            "correlator_coeff | {} correlator_creator | {} correlator_annihilator".format(
                site_a,
                site_b,
            ),
        )

    def get_particle_number_operator(self) -> OperatorSpecification:
        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {"particle_number_coeff": 1.0},
            {"particle_number": self.grid.get_x()},
            [
                f"particle_number_coeff | {i + 1} particle_number"
                for i in range(self.parameters.sites)
            ],
        )

    def get_site_occupation_operator(self, site_index: int) -> OperatorSpecification:
        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {"site_occupation_coeff": 1.0},
            {"site_occupation": self.grid.get_x()},
            f"site_occupation_coeff | {site_index + 1} site_occupation",
        )

    def get_site_occupation_operator_squared(
        self,
        site_index: int,
    ) -> OperatorSpecification:
        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {"site_occupation_squared_coeff": 1.0},
            {"site_occupation_squared": self.grid.get_x() ** 2},
            "site_occupation_squared_coeff | {} site_occupation_squared".format(
                site_index + 1,
            ),
        )

    def get_site_occupation_pair_operator(
        self,
        site_a: int,
        site_b: int,
    ) -> OperatorSpecification:
        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {"occupation_pair_coeff": 1.0},
            {"site_occupation": self.grid.get_x()},
            "occupation_pair_coeff | {} site_occupation | {} site_occupation".format(
                site_a,
                site_b,
            ),
        )

    def get_penalty_term(self, penalty: float) -> OperatorSpecification:
        table: List[str] = ["penalty_coeff_lambda_N^2 | 1 penalty_unity"]
        for i in range(self.parameters.sites):
            table.append(f"penalty_coeff_lambda | {i + 1} penalty_n_squared")
            table.append(f"penalty_coeff_m2_lambda_N | {i + 1} penalty_n")
            for j in range(i):
                table.append(
                    "penalty_coeff_2_lambda | {} penalty_n | {} penalty_n".format(
                        i + 1,
                        j + 1,
                    ),
                )

        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {
                "penalty_coeff_2_lambda": 2 * penalty,
                "penalty_coeff_lambda": penalty,
                "penalty_coeff_lambda_N^2": penalty * (self.parameters.N ** 2),
                "penalty_coeff_m2_lambda_N": -2 * penalty * self.parameters.N,
            },
            {
                "penalty_n": self.grid.get_x(),
                "penalty_n_squared": self.grid.get_x() ** 2,
                "penalty_unity": self.grid.get().get_unit_operator(),
            },
            table,
        )

    def get_efficient_penalty_term(
        self,
        gamma: float = 10.0,
        zeta: float = 0.0625,
    ) -> OperatorSpecification:
        N = self.parameters.N
        prefactor_1 = gamma / (zeta ** 2) * numpy.exp(-zeta * N)
        prefactor_2 = gamma / (zeta ** 2) * numpy.exp(zeta * N)
        constant = -2 * gamma / (zeta ** 2)

        term_1 = numpy.exp(zeta * self.grid.get_x())
        term_2 = numpy.exp(-zeta * self.grid.get_x())
        term_unit = self.grid.get().get_unit_operator()

        return OperatorSpecification(
            tuple(self.grid for i in range(self.parameters.sites)),
            {
                "penalty_coeff_1": prefactor_1,
                "penalty_coeff_2": prefactor_2,
                "penalty_coeff_3": constant,
            },
            {"penalty_1": term_1, "penalty_2": term_2, "penalty_3": term_unit},
            [
                "penalty_coeff_1 | "
                + " | ".join(
                    f"{i + 1} penalty_1" for i in range(self.parameters.sites)
                ),
                "penalty_coeff_2 | "
                + " | ".join(
                    f"{i + 1} penalty_2" for i in range(self.parameters.sites)
                ),
                "penalty_coeff_3 | 1 penalty_3",
            ],
        )

    @staticmethod
    @abstractmethod
    def create_parameters() -> Parameters:
        pass
