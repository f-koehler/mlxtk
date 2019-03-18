from typing import Any, Callable, Dict, List

import numpy

from ..doit_compat import DoitAction
from ..inout.expval import read_expval_ascii, write_expval_ascii
from .task import Task


class ComputeVariance(Task):
    def __init__(self, expectation_value: str, expectation_value_squared: str,
                 **kwargs):
        self.name = kwargs.get("name", expectation_value)
        self.expectation_value = expectation_value
        self.expectation_value_squared = expectation_value_squared

        self.path_expectation_value = self.expectation_value + ".exp"
        self.path_expectation_value_squared = self.expectation_value_squared + ".exp"
        self.path_variance = self.name + ".var"

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            t1, exp = read_expval_ascii(self.path_expectation_value)
            t2, exp2 = read_expval_ascii(self.path_expectation_value_squared)

            if not numpy.allclose(t1, t2):
                raise ValueError("got different time points")

            write_expval_ascii(self.path_variance, (t1, exp2 - (exp**2)))

        return {
            "name":
            "variance:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [self.path_variance],
            "file_dep":
            [self.path_expectation_value, self.path_expectation_value_squared],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_compute]
