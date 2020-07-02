import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Union

from mlxtk.doit_compat import DoitAction
from mlxtk.dvr import DVRSpecification
from mlxtk.hashing import inaccurate_hash
from mlxtk.log import get_logger
from mlxtk.operator import MBOperatorSpecification
from mlxtk.tasks.task import Task
from QDTK.Operatorb import OCoef as Coeff
from QDTK.Operatorb import Operatorb as Operator
from QDTK.Operatorb import OTerm as Term


class CreateMBOperator(Task):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.logger = get_logger(__name__ + ".CreateMBOperator")

        if "specification" in kwargs:
            self.specification = kwargs["specification"]
        else:
            if isinstance(args[0], MBOperatorSpecification):
                self.specification = args[0]
            else:
                self.specification = MBOperatorSpecification(*args, **kwargs)

        self.path = Path(self.name + ".mb_opr")
        self.path_pickle = Path(self.name + ".mb_opr_pickle")

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            obj = [
                self.name,
                self.specification.dofs,
                self.specification.grids,
                self.specification.coefficients,
                {},
                self.specification.table,
            ]
            for term in self.specification.terms:
                if isinstance(self.specification.terms[term], dict):
                    obj[4][term] = {
                        key: self.specification.terms[key]
                        for key in self.specification.terms
                    }
                    if "value" in obj[4][term]:
                        obj[4][term]["value"] = inaccurate_hash(obj[4][term]["value"])
                else:
                    obj[4][term] = inaccurate_hash(self.specification.terms[term])

            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp, protocol=3)

        return {
            "name": "mb_operator:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_write_operator(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_operator(targets: List[str]):
            del targets

            op = self.specification.get_operator()
            op.createOperatorFileb(str(self.path))

        return {
            "name": "mb_operator:{}:create".format(self.name),
            "actions": [action_write_operator],
            "targets": [self.path],
            "file_dep": [self.path_pickle],
        }

    def task_remove_operator(self) -> Dict[str, Any]:
        @DoitAction
        def action_remove_operator(targets: List[str]):
            del targets

            if self.path.exists():
                self.path.unlink()

        return {
            "name": "mb_operator:{}:remove".format(self.name),
            "actions": [action_remove_operator],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_operator]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_remove_operator]
