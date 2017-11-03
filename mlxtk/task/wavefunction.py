import os

from mlxtk.stringio import StringIO
from mlxtk.task import task


class WaveFunctionCreationTask(task.Task):
    def __init__(self, name, function, **kwargs):
        self.wave_function_name = name
        self.wave_function_creator = function

        kwargs["task_type"] = "WaveFunctionCreationTask"

        task.Task.__init__(
            self,
            "create_wave_function_file_" + name,
            self.write_wave_function_file,
            inputs=[
                task.FunctionInput("wave_function_string",
                                   self.get_wave_function_string)
            ],
            outputs=[task.FileOutput(name, name + ".wave_function")],
            **kwargs)

    def get_wave_function_string(self):
        wave_function = self.wave_function_creator()
        sio = StringIO()
        wave_function.createWfnFile(sio)
        return sio.getvalue()

    def write_wave_function_file(self):
        wave_function = self.wave_function_creator()
        path = os.path.join(self.cwd,
                            self.wave_function_name + ".wave_function")
        with open(path, "w") as fhandle:
            wave_function.createWfnFile(fhandle)
