import os

from mlxtk import hashing
from mlxtk.stringio import StringIO

from QDTK.Operator import Operator
from QDTK.Operatorb import Operatorb


class OperatorCreationTask(object):
    def __init__(self, project, name, func):
        self.project = project
        self.name = name
        self.func = func
        self.operator_data = None
        self.logger = project.get_logger("operator")

    def execute(self):
        self.logger.info("task: create operator \"%s\"", self.name)

        if self.is_up_to_date():
            self._update_project(False)
            self.logger.info("already up-to-date")
            self.logger.info("done")
            return

        # create the operator file
        self.logger.info("write operator file")
        with open(self._get_output_path(), "w") as fh:
            fh.write(self.operator_data)

        # mark operator update
        self._update_project(True)
        self.logger.info("done")

    def is_up_to_date(self):
        self._check_conflicts()
        self._get_operator_data()

        if not self._is_file_present():
            return False

        # check if operator already up-to-date
        hash_current = hashing.hash_file(self._get_output_path())
        hash_new = hashing.hash_string(self.operator_data)
        if hash_current != hash_new:
            self.logger.info("hash mismatch")
            self.logger.debug("%s != %s", hash_current, hash_new)
            return False

        return True

    def set_project_targets(self):
        self._check_conflicts()
        self.project.operators[self.name] = {"path": self._get_output_path()}

    def _check_conflicts(self):
        if self.name in self.project.operators:
            self.logger.critical("operator \"%s\" already exists", self.name)
            raise RuntimeError("name conflict: operator \"{}\" already exists".
                               format(self.name))

    def _get_operator_data(self):
        sio = StringIO()
        operator = self.func()
        if isinstance(operator, Operator):
            operator.createOperatorFile(sio)
        elif isinstance(operator, Operatorb):
            operator.createOperatorFileb(sio)
        else:
            raise TypeError(
                "Unknown operator type \"{}\"".format(type(operator).__name__))
        self.operator_data = sio.getvalue()

    def _get_output_path(self):
        return os.path.join(self.project.get_operator_dir(), self.name)

    def _is_file_present(self):
        ret = os.path.exists(self._get_output_path())
        if not ret:
            self.logger.info("operator file does not exist")
        return ret

    def _update_project(self, updated):
        self.project.operators[self.name] = {
            "updated": updated,
            "path": self._get_output_path()
        }
