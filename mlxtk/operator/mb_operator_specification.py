from typing import Any, Dict, Iterable, List, Union

from mlxtk.dvr import DVRSpecification
from QDTK.Operatorb import OCoef as Coeff
from QDTK.Operatorb import Operatorb as Operator
from QDTK.Operatorb import OTerm as Term


class MBOperatorSpecification:
    def __init__(
        self,
        dofs: Iterable[int],
        grids: Iterable[DVRSpecification],
        coefficients: List[Any],
        terms: List[Any],
        table: Union[str, List[str]],
    ):
        self.dofs = dofs
        self.grids = grids
        self.coefficients = coefficients
        self.terms = terms

        if isinstance(table, str):
            self.table = [table]
        else:
            self.table = table

    def __str__(self):
        output: List[str] = ["Many-Body Operator:"]
        output.append("\tDoFs: " + str(self.dofs))
        output.append("\tGrids:")
        for i, grid in enumerate(self.grids):
            output.append("\t\t{}: {}".format(i, grid))
        output.append("\tCoefficients:")
        for i, coefficient in enumerate(self.coefficients):
            output.append(
                "\t\t{}: {} = {}".format(i, coefficient, self.coefficients[coefficient])
            )
        output.append("\tTerms:")
        for i, term in enumerate(self.terms):
            output.append("\t\t{}: {}".format(i, term))
        output.append("\tTable:")
        for line in self.table:
            output.append("\t\t" + line)
        return "\n".join(output)

    def __add__(self, other):
        cpy = MBOperatorSpecification(
            self.dofs, self.grids, self.coefficients, self.terms, self.table
        )
        cpy.__iadd__(other)
        return cpy

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, MBOperatorSpecification):
            raise RuntimeError(
                (
                    "other object must be of type "
                    "MBOperatorSpecification as well (not {})".format(
                        type(other).__name__
                    )
                )
            )

        if self.dofs != other.dofs:
            raise ValueError("dofs differ")

        if self.grids != other.grids:
            raise ValueError("grids differ")

        if not set(self.coefficients.keys()).isdisjoint(set(other.coefficients.keys())):
            raise ValueError(
                "coefficient names are not unique\n"
                + str(self.coefficients)
                + "\n"
                + str(other.coefficients)
            )

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
        cpy = MBOperatorSpecification(
            self.dofs, self.grids, self.coefficients, self.terms, self.table
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
        cpy = MBOperatorSpecification(
            self.dofs, self.grids, self.coefficients, self.terms, self.table
        )
        cpy.__itruediv__(other)
        return cpy

    def _add_term(self, op: Operator, name: str, term):
        if not isinstance(term, dict):
            op.addLabel(name, Term(term))
            return

        if "td_name" in term:
            self._add_td_term(op, name, term)
            return

        op.addLabel(name, Term(term["value"], fft=term.get("fft", False)))

    def _add_td_term(self, op: Operator, name: str, term: Dict):
        term_kwargs = {}
        term_kwargs["fft"] = term.get("fft", False)

        if term_kwargs.get("type", "diag") != "diag":
            raise NotImplementedError(
                "Only diagonal time-dependent terms are supported by QDTK"
            )
        term_kwargs["type"] = "diag"

        term_kwargs["tf_label"] = term["td_name"]
        term_kwargs["tf_switch"] = term.get("td_switch", [0])
        term_kwargs["td"] = True

        if "td_args" in term:
            term_kwargs["tf_args"] = term["td_args"]

        op.addLabel(name, Term(**term_kwargs))

    def get_operator(self) -> Operator:
        op = Operator()
        op.define_dofs_and_grids(self.dofs, [grid.get() for grid in self.grids])

        for coeff in self.coefficients:
            op.addLabel(coeff, Coeff(self.coefficients[coeff]))

        for term in self.terms:
            self._add_term(op, term, self.terms[term])

        op.readTableb("\n".join(self.table))

        return op
