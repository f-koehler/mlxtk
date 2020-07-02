from typing import Any, List, Tuple, Union

import numpy

from mlxtk.dvr import DVRSpecification
from mlxtk.tools.diagonalize import diagonalize_1b_operator
from mlxtk.tools.operator import get_operator_matrix
from QDTK.Operator import OCoef as Coeff
from QDTK.Operator import Operator
from QDTK.Operator import OTerm as Term


class OperatorSpecification:
    """Object used to specify how to construct an operator acting on degrees
    of freedom.
    """

    def __init__(
        self,
        dofs: List[DVRSpecification],
        coefficients: List[Any],
        terms: List[Any],
        table: Union[str, List[str]],
    ):
        self.dofs = dofs
        self.coefficients = coefficients
        self.terms = terms

        if isinstance(table, str):
            self.table = [table]
        else:
            self.table = table

    def __add__(self, other):
        if not isinstance(other, OperatorSpecification):
            raise RuntimeError(
                ("other object must be of type " "OperatorSpecification as well")
            )
        cpy = OperatorSpecification(
            self.dofs, self.coefficients, self.terms, self.table
        )
        cpy.__iadd__(other)
        return cpy

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, OperatorSpecification):
            raise RuntimeError(
                ("other object must be of type " "OperatorSpecification as well")
            )

        if self.dofs != other.dofs:
            raise ValueError("dofs differ")

        if not set(self.coefficients.keys()).isdisjoint(set(other.coefficients.keys())):
            raise ValueError("coefficient names are not unique")

        if not set(self.terms.keys()).isdisjoint(set(other.terms.keys())):
            raise ValueError("term names are not unique")

        self.coefficients = {**self.coefficients, **other.coefficients}
        self.terms = {**self.terms, **other.terms}
        self.table += other.table

        return self

    def __imul__(self, other):
        for name in self.coefficients:
            self.coefficients[name] *= other
        return self

    def __mul__(self, other):
        cpy = OperatorSpecification(
            self.dofs, self.coefficients, self.terms, self.table
        )
        cpy.__imul__(other)
        return cpy

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        for name in self.coefficients:
            self.coefficients[name] /= other
        return self

    def __truediv__(self, other):
        cpy = OperatorSpecification(
            self.dofs, self.coefficients, self.terms, self.table
        )
        cpy.__itruediv__(other)
        return cpy

    def get_operator(self) -> Operator:
        op = Operator()
        op.define_grids([dof.get() for dof in self.dofs])

        for coeff in self.coefficients:
            op.addLabel(coeff, Coeff(self.coefficients[coeff]))

        for term in self.terms:
            op.addLabel(term, Term(self.terms[term]))

        op.readTable("\n".join(self.table))

        return op

    def get_matrix(self) -> numpy.ndarray:
        return get_operator_matrix(self.get_operator())

    def diagonalize(
        self, number_eigenfunctions: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        evals, evecs = diagonalize_1b_operator(self.get_matrix(), number_eigenfunctions)
        return evals, numpy.array(evecs)
