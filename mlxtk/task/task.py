import json
import os

from mlxtk import hashing
from mlxtk import log


class FileInput(object):
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.cwd = None

    def get_state(self):
        path = os.path.join(self.cwd, self.filename)
        if not os.path.exists(path):
            return None
        return hashing.hash_file(path)


class FunctionInput(object):
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.cwd = None

    def get_state(self):
        return hashing.hash_values(self.function())


class FileOutput(object):
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.cwd = None

    def get_state(self):
        path = os.path.join(self.cwd, self.filename)
        if not os.path.exists(path):
            return None
        return hashing.hash_file(path)

    def make_directories(self):
        dirname = os.path.dirname(self.filename)
        if dirname:
            if os.path.exists(dirname):
                return
            os.makedirs(os.path.join(self.cwd, dirname))


class Task(object):
    def __init__(self, name, function, **kwargs):
        self.name = name
        self.function = function
        self.inputs = kwargs.get("inputs", [])
        self.outputs = kwargs.get("outputs", [])
        self.cwd = kwargs.get("cwd", os.getcwd())
        self.task_type = kwargs.get("task_type", "Task")
        self.state_file = kwargs.get("state_file",
                                     os.path.join(self.cwd, name + ".state"))
        self.preprocess_steps = []
        self.input_states = None
        self.output_states = None
        self.stored_input_states = None
        self.stored_output_states = None

        self.logger = log.getLogger(self.task_type)

    def set_cwds(self):
        for inp in self.inputs:
            inp.cwd = self.cwd

        for out in self.outputs:
            out.cwd = self.cwd

    def get_current_state(self):
        for i, step in enumerate(self.preprocess_steps):
            self.logger.info("run preprocessing step %d", i)
            step()

        self.set_cwds()
        self.input_states = {inp.name: inp.get_state() for inp in self.inputs}
        self.output_states = {
            out.name: out.get_state()
            for out in self.outputs
        }

    def write_state_file(self):
        self.set_cwds()
        with open(self.state_file, "w") as fhandle:
            json.dump({
                "inputs": self.input_states,
                "outputs": self.output_states
            }, fhandle)

    def read_state_file(self):
        self.set_cwds()
        with open(self.state_file, "r") as fhandle:
            json_src = json.load(fhandle)

        self.stored_input_states = json_src["inputs"]
        self.stored_output_states = json_src["outputs"]

    def is_up_to_date(self):
        self.set_cwds()

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
                return False

        for out in self.outputs:
            if self.output_states[out.name] != self.stored_output_states[out.
                                                                         name]:
                self.logger.info("not up-to-date, input \"%s\" changed",
                                 out.name)
                return False

        return True

    def run(self):
        self.logger.info("enter task \"%s\"", self.name)

        if self.is_up_to_date():
            self.logger.info("already up-to-date")
            return False

        self.logger.info("create directories")
        for out in self.outputs:
            out.make_directories()

        self.logger.info("run task")
        self.function()

        self.logger.info("write state file")
        self.get_current_state()
        self.write_state_file()
