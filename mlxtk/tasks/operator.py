"""Create operators acting on distinguishable degrees of freedom.
"""
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import h5py
import numpy

from mlxtk.doit_compat import DoitAction
from mlxtk.dvr import DVRSpecification
from mlxtk.hashing import inaccurate_hash
from mlxtk.operator import OperatorSpecification
from mlxtk.tasks.task import Task
from mlxtk.tools.operator import get_operator_matrix
from QDTK.Operator import OCoef as Coeff
from QDTK.Operator import Operator
from QDTK.Operator import OTerm as Term


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
        self.path_matrix = Path(name + ".opr_mat.h5")
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
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": "operator:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
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
                dset = fptr.create_dataset(
                    "matrix", matrix.shape, dtype=numpy.complex128, compression="gzip"
                )
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
            "actions": [action_remove_operator],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_operator]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_remove_operator]
