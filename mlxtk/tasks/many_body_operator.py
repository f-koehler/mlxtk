import gzip
import io
import pickle
from typing import Any, Callable, Dict, Iterable, List, Union

from QDTK.Operatorb import OCoef as Coeff
from QDTK.Operatorb import Operatorb as Operator
from QDTK.Operatorb import OTerm as Term

from ..hashing import inaccurate_hash


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

    def __add__(self, other):
        cpy = MBOperatorSpecification(
            self.dofs, self.grids, self.coefficients, self.terms, self.table
        )
        cpy.__iadd__(other)
        return cpy

    def __radd(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if self.dofs != other.dofs:
            raise ValueError("dofs differ")

        if self.grids != other.grids:
            raise ValueError("grids differ")

        if not set(self.coefficients.keys()).isdisjoint(set(other.coefficients.keys())):
            raise ValueError("coefficient names are not unique")

        if not set(self.terms.keys()).isdisjoint(set(other.terms.keys())):
            raise ValueError("term names are not unique")

        self.coefficients = {**self.coefficients, **other.coefficients}
        self.terms = {**self.terms, **other.terms}
        self.table += other.table

        return self

    def _add_term(self, op: Operator, name: str, term):
        if not isinstance(term, dict):
            op.addLabel(name, term)
            return

        term_kwargs = {}
        term_kwargs["is_fft"] = term.get("fft", False)
        term_kwargs["type"] = term.get("type", "diag")

        if term_kwargs["type"] != diag:
            raise NotImplementedError(
                "Only diagonal time-dependent terms are supported by QDTK"
            )

        if "td_name" in term:
            term_kwargs["tf_label"] = term["td_name"]
            term_kwargs["tf_switch"] = term.get("td_switch", [0])

            if "td_args" in term:
                term_kwargs["tf_args"] = term["td_args"]

            op.addLabel(term, **term_kwargs)
        else:
            op.addLabel(term, term["value"], **kwargs)

    def get_operator(self):
        op = Operator()
        op.define_dofs_and_grids(dofs, [grid.get() for grid in self.grids])

        for coeff in self.coefficients:
            op.addLabel(coeff, Coeff(self.coefficients[coeff]))

        for term in self.terms:
            self._add_term(op, term, self.terms[term])

        op.readTableb("\n".join(self.table))

        return op


def create_many_body_operator(
    name: str, *args, **kwargs
) -> List[Callable[[], Dict[str, Any]]]:
    if "specification" in kwargs:
        return create_many_body_operator_impl(name, kwargs["specification"])

    if isinstance(args[0], MBOperatorSpecification):
        return create_many_body_operator_impl(name, args[0])

    return create_many_body_operator_impl(
        name, MBOperatorSpecification(*args, **kwargs)
    )


def create_many_body_operator_impl(
    name: str, specification: MBOperatorSpecification
) -> List[Callable[[], Dict[str, Any]]]:

    path_pickle = name + ".mb_opr_pickle"

    def task_write_parameters() -> Dict[str, Any]:
        def action_write_parameters(targets: List[str]):
            obj = [
                name,
                specification.dofs,
                specification.grids,
                specification.coefficients,
                {},
                specification.table,
            ]
            for term in specification.terms:
                if isinstance(terms[term], dict):
                    obj[4][term] = {
                        key: specification.terms[key] for key in specification.terms
                    }
                    if "value" in obj[4][term]:
                        obj[4][term]["value"] = inaccurate_hash(obj[4][term]["value"])
                else:
                    obj[4][term] = inaccurate_hash(specification.terms[term])

            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "create_many_body_operator:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_write_operator() -> Dict[str, Any]:
        path = name + ".mb_opr.gz"

        def action_write_operator(targets: List[str]):
            op = specification.get_operator()

            with gzip.open(targets[0], "wb") as fp:
                with io.StringIO() as sio:
                    op.createOperatorFileb(sio)
                    fp.write(sio.getvalue().encode())

        return {
            "name": "create_many_body_operator:{}:write_operator".format(name),
            "actions": [action_write_operator],
            "targets": [path],
            "file_dep": [path_pickle],
        }

    return [task_write_parameters, task_write_operator]
