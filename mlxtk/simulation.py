import os

from mlxtk import task


class Simulation(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.cwd = kwargs.get("cwd", name)
        self.tasks = []

        self.propagation_counter = 0
        self.relaxation_counter = 0

    def create_operator(self, name, function, **kwargs):
        self.tasks.append(task.OperatorCreationTask(name, function, **kwargs))
        self.tasks[-1].cwd = self.cwd

    def create_wave_function(self, name, function, **kwargs):
        self.tasks.append(
            task.WaveFunctionCreationTask(name, function, **kwargs))
        self.tasks[-1].cwd = self.cwd

    def propagate(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "propagation{}".format(self.propagation_counter)
            self.propagation_counter += 1

        self.tasks.append(
            task.PropagationTask(name, wave_function, operator, **kwargs))
        self.tasks[-1].cwd = self.cwd

    def relax(self, wave_function, operator, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
            kwargs.pop("name")
        else:
            name = "relaxation{}".format(self.relaxation_counter)
            self.relaxation_counter += 1

        kwargs["relax"] = True
        self.tasks.append(
            task.PropagationTask(name, wave_function, operator, **kwargs))
        self.tasks[-1].cwd = self.cwd

    def run(self):
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)

        for task in self.tasks:
            task.run()
