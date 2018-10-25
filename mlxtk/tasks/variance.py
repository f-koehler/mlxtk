import os
from typing import List

import numpy

from ..inout.expval import read_expval, write_expval_hdf5


def compute_variance(exp_op: str, exp_op2: str, **kwargs):
    if "name" in kwargs:
        name = kwargs["name"]
    else:
        name = exp_op

    def task_compute():
        def action_compute(targets: List[str]):
            del targets

            t1, exp = read_expval(exp_op + ".exp.hdf5")
            t2, exp2 = read_expval(exp_op2 + ".exp.hdf5")

            if not numpy.allclose(t1, t2):
                raise ValueError("got different time points")

            write_expval_hdf5(name + ".var.hdf5", (t1, exp2 - (exp ** 2)))

        return {
            "name": "variance:{}:compute".format(name),
            "actions": [action_compute],
            "targets": [name + ".var.hdf5"],
            "file_dep": [exp_op + ".exp.hdf5", exp_op2 + ".exp.hdf5"],
        }

    return [task_compute]
