import itertools
import os
import pickle


def Scan(object):
    def __init__(self, name, project_generator, **kwargs):
        self.name = name
        self.project_generator = project_generator
        self.cwd = "cwd"

        self.parameter_names = []
        self.parameter_indices = []
        self.parameter_values = []

        self.index_table = None
        self.value_table = None

    def add_static_parameter(self, name, values):
        pass

    def add_parameter(self, name, values):
        if name in self.parameter_names:
            raise RuntimeError("Duplicate parameter named \"{}\"".format(name))

        self.parameter_names.append(name)
        self.parameter_values.append(values)
        self.parameter_indices.append(list(range(0, len(values))))

    def init_tables(self):
        if self.index_table is None:
            self.index_table = list(
                itertools.product(*[para.indices for para in self.parameters]))

        if self.value_table is None:
            self.value_table = list(
                itertools.product(*[para.values for para in self.parameters]))

    def write_table(self):
        self.init_tables()

        tables = {
            "names": self.parameter_names,
            "indices": self.parameter_indices,
            "values": self.parameter_values
        }

        pickle_file = os.path.join(self.cwd, "parameters.pickle")
        with open(pickle_file, "wb") as fhandle:
            pickle.dump(fhandle, tables)
