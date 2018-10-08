import gzip
import io
import pickle
from typing import Any, Callable, Dict, List

import h5py
import numpy

from QDTK.Operator import OCoef as Coeff
from QDTK.Operator import Operator
from QDTK.Operator import OTerm as Term

from ..tools.operator import get_operator_matrix


def create_operator(name, dofs, coefficients, terms, table):
    if not isinstance(table, str):
        table = "\n".join(table)

    path_pickle = name + ".opr_pickle"

    def task_write_parameters():
        def action_write_parameters(targets):
            obj = [name, dofs, coefficients, terms, table]

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
            op = Operator()
            op.define_grids([dof.get() for dof in dofs])

            for coeff in coefficients:
                op.addLabel(coeff, Coeff(coefficients[coeff]))

            for term in terms:
                op.addLabel(term, Term(terms[term]))

            op.readTable(table)

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

                for i, dof in enumerate(dofs):
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
