"""
Todo:
    * Implement interaction properly.
"""
from mlxtk import tasks
from mlxtk.parameters import Parameters

from . import harmonic_trap


class HarmonicTrapHim(harmonic_trap.HarmonicTrap):
    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters([
            ("N", 2, "number of particles"),
            ("m", 5, "number of single particle functions"),
            ("g", 0.1,
             "strength of the harmonic interaction between the particles"),
            ("omega", 1.0, "angular frequency of the harmonic trap"),
        ])

    def get_interaction_operator(self) -> tasks.MBOperatorSpecification:
        return tasks.MBOperatorSpecification(
            (1, ),
            (self.grid, ),
            {"interaction_coeff": 2. * self.parameters.g},
            {"x": self.grid.get_x()},
            "interaction_coeff | 1 x | 1* x",
        )
