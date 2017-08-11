from QDTK.Operator import Operator
from QDTK.Operatorb import Operatorb

from mlxtk import hashing
from mlxtk import log

import os
import sys
if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO


class OperatorCreationTask:
    def __init__(self, project, name, func):
        self.project = project
        self.name = name
        self.func = func
        self.output_path = os.path.join(project.root_dir, name + ".opr")
        self.operator_data = None
        self.logger = log.getLogger("operator")

    def is_up_to_date(self):
        # check for name conflict
        if self.name in self.project.operators:
            self.logger.critical("operator \"%s\" already exists", self.name)
            raise RuntimeError("name conflict: operator \"{}\" already exists".
                               format(self.name))

        # calculate the operator and get the contents for the new operator file
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

        # check if operator already up-to-date
        if os.path.exists(self.output_path):
            hash_current = hashing.hash_file(self.output_path)
            hash_new = hashing.hash_string(self.operator_data)
            if hash_current != hash_new:
                self.logger.info("hash mismatch")
                self.logger.debug("  %s != %s", hash_current, hash_new)
                return False
        else:
            self.logger.info("file does not exist")
            return False

        return True

    def execute(self):
        self.logger.info("task: create operator \"%s\"", self.name)

        if self.is_up_to_date():
            self.project.operators[self.name] = {
                "updated": False,
                "path": self.output_path
            }
            self.logger.info("already up-to-date")
            self.logger.info("done")
            return

        # create the operator file
        self.logger.info("write operator file")
        with open(self.output_path, "w") as fh:
            fh.write(self.operator_data)

        # mark operator update
        self.project.operators[self.name] = {
            "updated": False,
            "path": self.output_path
        }

        self.logger.info("done")
