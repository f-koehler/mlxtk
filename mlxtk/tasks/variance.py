from typing import List

import numpy

from ..doit_compat import DoitAction
from ..inout.expval import read_expval_ascii, write_expval_ascii


def compute_variance(exp_op: str, exp_op2: str, **kwargs):
    name = kwargs.get("name", exp_op)

    def task_compute():
        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            t1, exp = read_expval_ascii(exp_op + ".exp")
            t2, exp2 = read_expval_ascii(exp_op2 + ".exp")

            if not numpy.allclose(t1, t2):
                raise ValueError("got different time points")

            write_expval_ascii(name + ".var", (t1, exp2 - (exp**2)))

        return {
            "name": "variance:{}:compute".format(name),
            "actions": [action_compute],
            "targets": [name + ".var"],
            "file_dep": [exp_op + ".exp", exp_op2 + ".exp"],
        }

    return [task_compute]
