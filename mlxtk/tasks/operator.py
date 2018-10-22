import gzip
import io
import pickle
from typing import Any, Callable, Dict, Iterable, List, Union

import h5py
import numpy

from QDTK.Operator import OCoef as Coeff
from QDTK.Operator import Operator
from QDTK.Operator import OTerm as Term

from ..dvr import DVRSpecification
from ..hashing import inaccurate_hash
from ..tools.operator import get_operator_matrix


class OperatorSpecification:
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
        cpy = OperatorSpecification(
            self.dofs, self.coefficients, self.terms, self.table
        )
        cpy.__iadd__(other)
        return cpy

    def __radd(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
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

    def get_operator(self):
        op = Operator()
        op.define_grids([dof.get() for dof in self.dofs])

        for coeff in self.coefficients:
            op.addLabel(coeff, Coeff(self.coefficients[coeff]))

        for term in self.terms:
            op.addLabel(term, Term(self.terms[term]))

        op.readTable("\n".join(self.table))

        return op


def create_operator(name: str, *args, **kwargs):
    if "specification" in kwargs:
        return create_operator_impl(name, kwargs["specification"])

    if isinstance(args[0], OperatorSpecification):
        return create_operator_impl(name, args[0])

    return create_operator_impl(name, OperatorSpecification(*args, **kwargs))


def create_operator_impl(
    name: str, specification: OperatorSpecification
) -> List[Callable[[], Dict[str, Any]]]:
    path_pickle = name + ".opr_pickle"

    def task_write_parameters():
        def action_write_parameters(targets):
            obj = [
                name,
                specification.dofs,
                specification.coefficients,
                {},
                specification.table,
            ]
            for term in specification.terms:
                obj[3][term] = inaccurate_hash(specification.terms[term])

            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "create_operator:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_write_operator():
        path = name + ".opr.gz"
        path_matrix = name + ".opr_mat.hdf5"

        def action_write_operator(targets):
            op = specification.get_operator()

            with gzip.open(targets[0], "wb") as fp:
                with io.StringIO() as sio:
                    op.createOperatorFile(sio)
                    fp.write(sio.getvalue().encode())

            matrix = get_operator_matrix(op)
            with h5py.File(path_matrix, "w") as fp:
                dset = fp.create_dataset(
                    "matrix", matrix.shape, dtype=numpy.complex128, compression="gzip"
                )
                dset[:, :] = matrix

                for i, dof in enumerate(specification.dofs):
                    grid = dof.get_x()
                    dset = fp.create_dataset(
                        "grid_{}".format(i + 1),
                        grid.shape,
                        dtype=numpy.float64,
                        compression="gzip",
                    )
                    dset[:] = grid

                    weights = dof.get_weights()
                    dset = fp.create_dataset(
                        "weights_{}".format(i + 1),
                        grid.shape,
                        dtype=numpy.float64,
                        compression="gzip",
                    )
                    dset[:] = weights

        return {
            "name": "create_operator:{}:write_operator".format(name),
            "actions": [action_write_operator],
            "targets": [path, path_matrix],
            "file_dep": [path_pickle],
        }

    return [task_write_parameters, task_write_operator]
