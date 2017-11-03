import os

from QDTK.Operator import Operator
from QDTK.Operatorb import Operatorb

from mlxtk.stringio import StringIO
from mlxtk.task import task


class OperatorCreationTask(task.Task):
    def __init__(self, name, function, **kwargs):
        self.operator_name = name
        self.operator_creator = function

        kwargs["task_type"] = "OperatorCreationTask"

        task.Task.__init__(
            self,
            "create_operator_file_" + name,
            self.write_operator_file,
            inputs=[task.FunctionInput("operator_string", self.get_operator_string)],
            outputs=[task.FileOutput(name, name + ".operator")],
            **kwargs)

    def get_operator_string(self):
        operator = self.operator_creator()
        sio = StringIO()
        if isinstance(operator, Operator):
            operator.createOperatorFile(sio)
        elif isinstance(operator, Operatorb):
            operator.createOperatorFileb(sio)
        else:
            raise TypeError("Unknown operator type \"{}\"".format(
                type(operator).__name__))

        return sio.getvalue()

    def write_operator_file(self):
        operator = self.operator_creator()
        path = os.path.join(self.cwd, self.operator_name + ".operator")
        with open(path, "w") as fhandle:
            if isinstance(operator, Operator):
                operator.createOperatorFile(fhandle)
            elif isinstance(operator, Operatorb):
                operator.createOperatorFileb(fhandle)
            else:
                raise TypeError("Unknown operator type \"{}\"".format(
                    type(operator).__name__))
