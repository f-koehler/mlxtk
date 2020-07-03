from typing import Optional, Union

import numpy

from mlxtk.parameters import Parameters
from mlxtk.systems.bose_bose.bose_bose import BoseBoseSystem
from mlxtk.tasks import MBOperatorSpecification, OperatorSpecification


def gaussian(x: Union[float, numpy.ndarray], x0: float, alpha: float = 1.0):
    return numpy.exp(-((x - x0) ** 2) / (alpha ** 2))


class TwoGaussianTraps(BoseBoseSystem):
    @staticmethod
    def create_parameters():
        return Parameters(
            [
                ("N_A", 2, "number of A particles"),
                ("N_B", 2, "number of B particles"),
                ("M_A", 5, "number of SBSs (A species)"),
                ("M_B", 5, "number of SBSs (B species)"),
                ("m_A", 5, "number of SPFs (A species)"),
                ("m_B", 5, "number of SPFs (B species)"),
                ("mass_A", 1.0, "mass of A species particles"),
                ("mass_B", 1.0, "mass of B species particles"),
                ("g_AA", 0.1, "intra-species interaction (A species)"),
                ("g_BB", 0.1, "intra-species interaction (B species)"),
                ("g_AB", 1.0, "inter-species interaction"),
                ("V0L", 1.0, "depth of the left Gaussian well"),
                ("V0R", 1.0, "depth of the right Gaussian well"),
                ("x0L", -3.5, "center of the left Gaussian well"),
                ("x0R", 3.5, "center of the right Gaussian well"),
                ("alpha", 1.0, "well asymmetry"),
            ]
        )

    def get_potential_operator_left_1b_A(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"potential_left_coeff_A": -self.parameters.V0L},
            {
                "potential_left_A": gaussian(
                    self.grid.get_x(), self.parameters.x0L, 1.0
                ),
            },
            "potential_left_coeff_A | 1 potential_left_A",
        )

    def get_potential_operator_left_1b_B(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"potential_left_coeff_A": -self.parameters.V0L},
            {
                "potential_left_A": gaussian(
                    self.grid.get_x(), self.parameters.x0L, 1.0
                ),
            },
            "potential_left_coeff_A | 1 potential_left_A",
        )

    def get_potential_operator_right_1b_A(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"potential_right_coeff_A": -self.parameters.V0R},
            {
                "potential_right_A": gaussian(
                    self.grid.get_x(), self.parameters.x0R, self.parameters.alpha
                ),
            },
            "potential_right_coeff_A | 1 potential_right_A",
        )

    def get_potential_operator_right_1b_B(self) -> OperatorSpecification:
        return OperatorSpecification(
            (self.grid_1b,),
            {"potential_right_coeff_B": -self.parameters.V0R},
            {
                "potential_right_B": gaussian(
                    self.grid.get_x(), self.parameters.x0R, self.parameters.alpha
                ),
            },
            "potential_right_coeff_B | 1 potential_right_B",
        )

    def get_potential_operator_1b_A(self) -> OperatorSpecification:
        return (
            self.get_potential_operator_left_1b_A()
            + self.get_potential_operator_left_1b_A()
        )

    def get_potential_operator_1b_B(self) -> OperatorSpecification:
        return (
            self.get_potential_operator_left_1b_B()
            + self.get_potential_operator_left_1b_B()
        )

    def get_potential_operator_left_A(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"potential_left_coeff_A": -self.parameters.V0L},
            {"potential_left_A": gaussian(self.grid.get_x(), self.parameters.x0L, 1.0)},
            "potential_left_coeff_A | 1 potential_left_A",
        )

    def get_potential_operator_left_B(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"potential_left_coeff_B": -self.parameters.V0L},
            {"potential_left_B": gaussian(self.grid.get_x(), self.parameters.x0L, 1.0)},
            "potential_left_coeff_B | 2 potential_left_B",
        )

    def get_potential_operator_right_A(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"potential_right_coeff_A": -self.parameters.V0R},
            {
                "potential_right_A": gaussian(
                    self.grid.get_x(), self.parameters.x0R, self.parameters.alpha
                )
            },
            "potential_right_coeff_A | 1 potential_right_A",
        )

    def get_potential_operator_right_B(self) -> MBOperatorSpecification:
        return MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"potential_right_coeff_B": -self.parameters.V0R},
            {
                "potential_right_B": gaussian(
                    self.grid.get_x(), self.parameters.x0R, self.parameters.alpha
                )
            },
            "potential_right_coeff_B | 2 potential_right_B",
        )

    def get_potential_operator_left(self) -> MBOperatorSpecification:
        return (
            self.get_potential_operator_left_A() + self.get_potential_operator_left_B()
        )

    def get_hamiltonian_left_well_1b_A(self) -> OperatorSpecification:
        return (
            self.get_kinetic_operator_1b_A() + self.get_potential_operator_left_1b_A()
        )

    def get_hamiltonian_left_well_1b_B(self) -> OperatorSpecification:
        return (
            self.get_kinetic_operator_1b_B() + self.get_potential_operator_left_1b_B()
        )

    def get_hamiltonian_right_well_1b_A(self) -> OperatorSpecification:
        return (
            self.get_kinetic_operator_1b_A() + self.get_potential_operator_right_1b_A()
        )

    def get_hamiltonian_right_well_1b_B(self) -> OperatorSpecification:
        return (
            self.get_kinetic_operator_1b_B() + self.get_potential_operator_right_1b_B()
        )

    def get_potential_operator_right(self) -> MBOperatorSpecification:
        return (
            self.get_potential_operator_right_A()
            + self.get_potential_operator_right_B()
        )

    def get_potential_operator(self) -> MBOperatorSpecification:
        return self.get_potential_operator_left() + self.get_potential_operator_right()

    def get_hamiltonian(self) -> MBOperatorSpecification:
        operator = self.get_kinetic_operator() + self.get_potential_operator()
        if self.parameters.g_AB != 0.0:
            operator += self.get_interaction_operator_AB()

        if (self.parameters.N_A > 1) and (self.parameters.g_AA != 0.0):
            operator += self.get_interaction_operator_AA()

        if (self.parameters.N_B > 1) and (self.parameters.g_BB != 0.0):
            operator += self.get_interaction_operator_BB()

        return operator

    def get_hamiltonian_left_right(self) -> MBOperatorSpecification:
        operator = (
            self.get_kinetic_operator()
            + self.get_potential_operator_left_A()
            + self.get_potential_operator_right_B()
        )
        if self.parameters.g_AB != 0.0:
            operator += self.get_interaction_operator_AB()

        if (self.parameters.N_A > 1) and (self.parameters.g_AA != 0.0):
            operator += self.get_interaction_operator_AA()

        if (self.parameters.N_B > 1) and (self.parameters.g_BB != 0.0):
            operator += self.get_interaction_operator_BB()

        return operator

    def get_hamiltonian_colliding(
        self,
        vL: float = 1.0,
        aL: float = 0.0,
        vR: Optional[float] = None,
        aR: Optional[float] = None,
    ) -> MBOperatorSpecification:
        if vR is None:
            vR = -vL

        if aR is None:
            aR = -aL

        left_potential = MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"potential_left_coeff_A": 1.0, "potential_left_coeff_B": 1.0},
            {
                "potential_left_A": {
                    "td_name": "moving_gaussian",
                    "td_args": [-self.parameters.V0L, self.parameters.x0L, vL, aL],
                },
                "potential_left_B": {
                    "td_name": "moving_gaussian",
                    "td_args": [-self.parameters.V0L, self.parameters.x0L, vL, aL],
                },
            },
            [
                "potential_left_coeff_A | 1 potential_left_A",
                "potential_left_coeff_B | 2 potential_left_B",
            ],
        )

        right_potential = MBOperatorSpecification(
            (1, 1),
            (self.grid, self.grid),
            {"potential_right_coeff_A": 1.0, "potential_right_coeff_B": 1.0},
            {
                "potential_right_A": {
                    "td_name": "moving_gaussian",
                    "td_args": [-self.parameters.V0R, self.parameters.x0R, vR, aR],
                },
                "potential_right_B": {
                    "td_name": "moving_gaussian",
                    "td_args": [-self.parameters.V0R, self.parameters.x0R, vR, aR],
                },
            },
            [
                "potential_right_coeff_A | 1 potential_right_A",
                "potential_right_coeff_B | 2 potential_right_B",
            ],
        )

        operator = self.get_kinetic_operator() + left_potential + right_potential
        if self.parameters.g_AB != 0.0:
            operator += self.get_interaction_operator_AB()

        if (self.parameters.N_A > 1) and (self.parameters.g_AA != 0.0):
            operator += self.get_interaction_operator_AA()

        if (self.parameters.N_B > 1) and (self.parameters.g_BB != 0.0):
            operator += self.get_interaction_operator_BB()

        return operator
