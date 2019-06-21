"""Create operators acting on distinguishable degrees of freedom.
"""
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import h5py
import numpy

from QDTK.Operator import OCoef as Coeff
from QDTK.Operator import Operator
from QDTK.Operator import OTerm as Term

from ..doit_compat import DoitAction
from ..dvr import DVRSpecification
from ..hashing import inaccurate_hash
from ..tools.operator import get_operator_matrix
from .task import Task


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
            raise RuntimeError(("other object must be of type "
                                "OperatorSpecification as well"))
        cpy = OperatorSpecification(self.dofs, self.coefficients, self.terms,
                                    self.table)
        cpy.__iadd__(other)
        return cpy

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, OperatorSpecification):
            raise RuntimeError(("other object must be of type "
                                "OperatorSpecification as well"))

        if self.dofs != other.dofs:
            raise ValueError("dofs differ")

        if not set(self.coefficients.keys()).isdisjoint(
                set(other.coefficients.keys())):
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
        cpy = OperatorSpecification(self.dofs, self.coefficients, self.terms,
                                    self.table)
        cpy.__imul__(other)
        return cpy

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        for name in self.coefficients:
            self.coefficients[name] /= other
        return self

    def __truediv__(self, other):
        cpy = OperatorSpecification(self.dofs, self.coefficients, self.terms,
                                    self.table)
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


class CreateOperator(Task):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

        if "specification" in kwargs:
            self.specification = kwargs["specification"]
        else:
            if isinstance(args[0], OperatorSpecification):
                self.specification = args[0]
            else:
                self.specification = OperatorSpecification(*args, **kwargs)

        self.path = Path(name + ".opr")
        self.path_matrix = Path(name + ".opr_mat.hdf5")
        self.path_pickle = Path(name + ".opr_pickle")

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [
                self.name,
                self.specification.dofs,
                self.specification.coefficients,
                {},
                self.specification.table,
            ]
            for term in self.specification.terms:
                obj[3][term] = inaccurate_hash(self.specification.terms[term])

            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr)

        return {
            "name": "operator:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle]
        }

    def task_write_operator(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_operator(targets: List[str]):
            del targets

            op = self.specification.get_operator()

            with open(self.path, "w") as fptr:
                op.createOperatorFile(fptr)

            matrix = get_operator_matrix(op)
            with h5py.File(self.path_matrix, "w") as fptr:
                dset = fptr.create_dataset("matrix",
                                           matrix.shape,
                                           dtype=numpy.complex128,
                                           compression="gzip")
                dset[:, :] = matrix

                for i, dof in enumerate(self.specification.dofs):
                    grid = dof.get_x()
                    dset = fptr.create_dataset(
                        "grid_{}".format(i + 1),
                        grid.shape,
                        dtype=numpy.float64,
                        compression="gzip",
                    )
                    dset[:] = grid

                    weights = dof.get_weights()
                    dset = fptr.create_dataset(
                        "weights_{}".format(i + 1),
                        grid.shape,
                        dtype=numpy.float64,
                        compression="gzip",
                    )
                    dset[:] = weights

        return {
            "name": "operator:{}:create".format(self.name),
            "actions": [action_write_operator],
            "targets": [self.path, self.path_matrix],
            "file_dep": [self.path_pickle],
        }

    def task_remove_operator(self) -> Dict[str, Any]:
        @DoitAction
        def action_remove_operator(targets: List[str]):
            del targets

            if self.path.exists():
                self.path.unlink()

            if self.path_matrix.exists():
                self.path_matrix.unlink()

        return {
            "name": "operator:{}:remove".format(self.name),
            "actions": [action_remove_operator]
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_operator]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_remove_operator]
