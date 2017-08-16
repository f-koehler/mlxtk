import itertools


def get_cartesian_product(*args):
    return [element for element in itertools.product(*args)]


class Parameter(object):
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.indices = [i for i in range(0, len(values))]


class ParameterScan(object):
    def __init__(self, func):
        self.parameters = []
        self.func = func

    def add_parameter(self, parameter):
        self.parameters.append(parameter)
