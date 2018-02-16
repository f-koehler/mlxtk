import os
import numpy

from mlxtk.stringio import StringIO
from mlxtk.task import task
from mlxtk.tools.wave_function_manipulation import load_wave_function


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
        sio = StringIO()

        # generate current wave function
        wave_function = self.wave_function_creator(self.parameters)

        # Due to numerical issues the components can marginally differ even if
        # the input did not change. If this is the case we pretend that the
        # task has not changed.
        path = self.wave_function_name + ".wave_function"
        if os.path.exists(path):
            stored_wave_function = load_wave_function(path)

            max_diff = numpy.max(
                numpy.abs(wave_function._psi) -
                numpy.abs(stored_wave_function._psi))

            if max_diff < 1e-9:
                self.logger.warn(
                    ("the maximal absolute difference to the stored wave "
                     "function is %s < 1e-9, the wave functions are considered"
                     " identical"), "{:e}".format(max_diff))
                stored_wave_function.createWfnFile(sio)
                return sio.getvalue()
            else:
                self.logger.warn(
                    ("the maximal absolute difference to the stored wave "
                     "function is %s >= 1e-9, the wave functions are not "
                     "considered identical"), "{:e}".format(max_diff))

        wave_function.createWfnFile(sio)
        return sio.getvalue()

    def write_wave_function_file(self):
        wave_function = self.wave_function_creator(self.parameters)
        path = self.wave_function_name + ".wave_function"
        with open(path, "w") as fhandle:
            wave_function.createWfnFile(fhandle)
