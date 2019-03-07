from mlxtk import tasks
from mlxtk.parameters import Parameters

from .gaussian_trap import GaussianTrap


class TwoGaussianTraps(GaussianTrap):
    @staticmethod
    def create_parameters():
        return Parameters([
            ("N", 2, "number of particles"),
            ("m", 5, "number of single particle functions"),
            ("V0L", 1.0, "depth of the left Gaussian well"),
            ("V0R", 1.0, "depth of the right Gaussian well"),
            ("x0L", 0.0, "center of the left Gaussian well"),
            ("x0R", 0.0, "center of the right Gaussian well"),
            ("alpha", 1.0, "well asymmetry"),
            ("g", 0.1, "strength of the contact interaction"),
        ])

    def get_potential_operator_1b(self) -> tasks.OperatorSpecification:
        return self.create_gaussian_potential_operator_1b(
            self.parameters.x0L, self.parameters.V0L,
            "potential_left") + self.create_gaussian_potential_operator_1b(
                self.parameters.x0R, self.parameters.V0R, "potential_right",
                self.parameters.alpha)

    def get_potential_operator(self) -> tasks.MBOperatorSpecification:
        return self.create_gaussian_potential_operator(
            self.parameters.x0L, self.parameters.V0L,
            "potential_left") + self.create_gaussian_potential_operator(
                self.parameters.x0R, self.parameters.V0R, "potential_right",
                self.parameters.alpha)
