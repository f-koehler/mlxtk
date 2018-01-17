import json
import os
import time

from mlxtk import hashing
from mlxtk import log


class FileInput(object):
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.logger = log.get_logger(__name__)

    def get_state(self):
        path = self.filename
        if not os.path.exists(path):
            return None
        return hashing.hash_file(path)


class FunctionInput(object):
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def get_state(self):
        return hashing.hash_values(self.function())


class FileOutput(object):
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename

    def get_state(self):
        path = self.filename
        if not os.path.exists(path):
            return None
        return hashing.hash_file(path)

    def make_directories(self):
        dirname = os.path.dirname(self.filename)
        if dirname:
            if os.path.exists(dirname):
                return
            os.makedirs(dirname)


class Task(object):
    def __init__(self, name, function, **kwargs):
        self.name = name
        self.function = function
        self.inputs = kwargs.get("inputs", [])
        self.outputs = kwargs.get("outputs", [])
        self.task_type = kwargs.get("task_type", "Task")
        self.state_file = os.path.join("states", name + ".json")
        self.preprocess_steps = []
        self.input_states = None
        self.output_states = None
        self.stored_input_states = None
        self.stored_output_states = None
        self.parameters = None

        self.logger = log.get_logger(__name__ + "(" + self.task_type + ")")

    def get_current_state(self):
        for i, step in enumerate(self.preprocess_steps):
            self.logger.info("run preprocessing step %d", i)
            step()

        self.input_states = {inp.name: inp.get_state() for inp in self.inputs}
        self.output_states = {
            out.name: out.get_state()
            for out in self.outputs
        }

    def write_state_file(self):
        with open(self.state_file, "w") as fhandle:
            json.dump({
                "inputs": self.input_states,
                "outputs": self.output_states
            }, fhandle)

    def read_state_file(self):
        with open(self.state_file, "r") as fhandle:
            json_src = json.load(fhandle)

        self.stored_input_states = json_src["inputs"]
        self.stored_output_states = json_src["outputs"]

    def is_up_to_date(self):
        """Check if the task is up-to-date.
        """

        if not os.path.exists(self.state_file):
            self.logger.info("not up-to-date, state file does not exist")
            return False

        self.get_current_state()
        self.read_state_file()

        if len(self.input_states) != len(self.stored_input_states):
            self.logger.info("not up-to-date, number of inputs changed")
            return False

        if len(self.output_states) != len(self.stored_output_states):
            self.logger.info("not up-to-date, number of outputs")
            return False

        for inp in self.inputs:
            if self.input_states[inp.name] != self.stored_input_states[inp.
                                                                       name]:
                self.logger.info("not up-to-date, input \"%s\" changed",
                                 inp.name)
                self.logger.debug("  current: <%s>",
                                  self.input_states[inp.name])
                self.logger.debug("  stored:  <%s>",
                                  self.stored_input_states[inp.name])
                return False

        for out in self.outputs:
            if self.output_states[out.name] is None:
                self.logger.info(
                    "not up-to-date, output \"%s\" does not exist", out.name)
                return False

        return True

    def run(self):
        """Execute the task.
        """
        self.logger.info("enter task \"%s\"", self.name)

        if self.is_up_to_date():
            self.logger.info("already up-to-date")
            return False

        self.logger.info("create directories")
        for out in self.outputs:
            out.make_directories()

        self.logger.info("run task")
        start = time.perf_counter()
        self.function()
        stop = time.perf_counter()
        self.logger.info("execution took %fs", stop - start)

        self.logger.info("write state file")
        self.get_current_state()
        self.write_state_file()
