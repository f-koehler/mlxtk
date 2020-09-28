import numpy

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import MBOperatorSpecification


class BoseHubbard:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.grid = dvr.add_sinedvr(parameters.sites, 0, parameters.sites - 1)
        self.logger = get_logger(__name__ + ".BoseHubbard")

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("sites", "4", "number of sites"),
                ("N", 4, "number of particles"),
                ("U", 1.0, "interaction strength in units of hopping constant"),
                ("pbc", True, "whether to use periodic boundary conditions"),
            ]
        )

    def create_hopping_term(self) -> MBOperatorSpecification:
        matrix = numpy.zeros((self.parameters.sites - 1, self.parameters.sites - 1))
        for i in range(self.parameters.sites - 1):
            matrix[i, i + 1] = 1.0
            matrix[i + 1, i] = 1.0

        if self.parameters.pbc:
            matrix[0, -1] = 1.0
            matrix[-1, 0] = 1.0

        return MBOperatorSpecification(
            (1,),
            (self.grid,),
            {
                "hopping_coeff": -1.0,
            },
            {"hopping": matrix},
            "hopping_coeff |1 hopping",
        )
