from abc import ABC, abstractmethod

import numpy

from mlxtk.dvr import DVRSpecification
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import MBOperatorSpecification, OperatorSpecification


class BoseBoseSystem(ABC):
    def __init__(
        self,
        parameters: Parameters,
        grid: DVRSpecification,
    ):
        self.parameters = parameters
        self.grid = grid
        self.logger = get_logger(__name__ + ".BoseBoseSystem")

        if grid.is_fft():
            self.grid_1b = grid.get_expdvr()
        else:
            self.grid_1b = self.grid

    @staticmethod
    @abstractmethod
    def create_parameters() -> Parameters:
        pass

    def get_kinetic_operator_1b_A(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"kinetic_coeff_A": -0.5 * self.parameters.mass_A},
            {"kinetic_A": self.grid_1b.get_d2()},
            "kinetic_coeff_A | 1 kinetic_A",
        )

    def get_kinetic_operator_1b_B(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"kinetic_coeff_B": -0.5 * self.parameters.mass_B},
            {"kinetic_B": self.grid_1b.get_d2()},
            "kinetic_coeff_B | 1 kinetic_B",
        )

    def get_kinetic_operator_A(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {
                "kinetic_coeff_A": -0.5 * self.parameters.mass_A,
            },
            {
                "kinetic_A": {
                    "value": self.grid.get_d2(),
                    "fft": self.grid.is_fft(),
                },
            },
            "kinetic_coeff_A | 1 kinetic_A",
        )

    def get_kinetic_operator_B(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {
                "kinetic_coeff_B": -0.5 * self.parameters.mass_B,
            },
            {
                "kinetic_B": {
                    "value": self.grid.get_d2(),
                    "fft": self.grid.is_fft(),
                },
            },
            "kinetic_coeff_B | 2 kinetic_B",
        )

    def get_kinetic_operator(self) -> MBOperatorSpecification:
        return self.get_kinetic_operator_A() + self.get_kinetic_operator_B()

    def get_interaction_operator_AA(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"interaction_coeff_AA": self.parameters.g_AA},
            {"interaction_AA": self.grid.get_delta()},
            ["interaction_coeff_AA | {1:1} interaction_AA"],
        )

    def get_interaction_operator_BB(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"interaction_coeff_BB": self.parameters.g_BB},
            {"interaction_BB": self.grid.get_delta()},
            ["interaction_coeff_BB | {2:2} interaction_BB"],
        )

    def get_interaction_operator_AB(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"interaction_coeff_AB": self.parameters.g_AB},
            {"interaction_AB": self.grid.get_delta()},
            ["interaction_coeff_AB | {1:2} interaction_AB"],
        )

    def get_com_operator_A(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"com_coeff_A": 1.0 / self.parameters.N_A},
            {"com_A": self.grid.get_x()},
            ["com_coeff_A | 1 com_A"],
        )

    def get_com_operator_B(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"com_coeff_B": 1.0 / self.parameters.N_B},
            {"com_B": self.grid.get_x()},
            ["com_coeff_B | 2 com_B"],
        )
