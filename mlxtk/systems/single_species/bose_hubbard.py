from typing import List

import numpy

from mlxtk import dvr
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters
from mlxtk.tasks import MBOperatorSpecification, OperatorSpecification


class BoseHubbard:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.grid = dvr.add_sinedvr(parameters.sites, 0, parameters.sites - 1)
        self.logger = get_logger(__name__ + ".BoseHubbard")

    @staticmethod
    def create_parameters() -> Parameters:
        return Parameters(
            [
                ("sites", 4, "number of sites"),
                ("N", 4, "number of particles"),
                ("m", 4, "number of SPFs"),
                ("J", 1.0, "hopping constant"),
                ("U", 1.0, "interaction strength"),
                ("pbc", True, "whether to use periodic boundary conditions"),
            ]
        )

    def create_hopping_term(self) -> MBOperatorSpecification:
        matrix = numpy.zeros((self.parameters.sites, self.parameters.sites))
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
                "hopping_coeff": -self.parameters.J,
            },
            {"hopping": matrix},
            "hopping_coeff | 1 hopping",
        )

    def create_interaction_term(self) -> MBOperatorSpecification:
        def create_delta_peak(n: int, i: int) -> numpy.ndarray:
            result = numpy.zeros(n)
            result[i] = 1.0
            return result

        n = self.grid.get().npoints

        return MBOperatorSpecification(
            (1,),
            (self.grid,),
            {"interaction_coeff": self.parameters.U},
            {
                "interaction_term_{}".format(i): create_delta_peak(n, i)
                for i in range(n)
            },
            [
                "interaction_coeff | 1 interaction_term_{} | 1* interaction_term_{}".format(
                    i, i
                )
                for i in range(n)
            ],
        )

    def get_hamiltonian(self) -> MBOperatorSpecification:
        terms: List[MBOperatorSpecification] = []
        if self.parameters.J != 0.0:
            terms.append(self.create_hopping_term())
        if self.parameters.U != 0.0:
            terms.append(self.create_interaction_term())
        if terms is None:
            raise RuntimeError("Hamiltonian would be empty since both J and U are 0.0")
        operator = terms[0]
        for term in terms[1:]:
            operator += term
        print(operator)
        return operator

    def get_fake_initial_state_hamiltonian(self) -> OperatorSpecification:
        n = self.grid.get().npoints
        mat = numpy.zeros((n, n), dtype=numpy.float64)
        for i in range(n):
            mat[i, i] = i

        return OperatorSpecification(
            (self.grid,),
            {
                "coeff": 1,
            },
            {
                "matrix": mat,
            },
            "coeff | 1 matrix",
        )

    def distribute_particles(self) -> numpy.ndarray:
        ns = numpy.zeros(self.parameters.m)
        loc = 0
        for _ in range(self.parameters.N):
            ns[loc] += 1
            loc = (loc + 1) % self.parameters.m
        assert numpy.sum(ns) == self.parameters.N
        return ns