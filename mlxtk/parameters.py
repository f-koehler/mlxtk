import copy
import functools
import itertools
import json
import numpy
import os
import pandas
import pickle

from . import tabulate


class Parameters(object):
    """A set of parameters for simulations and parameter scans

    If no dictionary is given in the construction, the parameter set will not
    contain any parameters initially.
    Parameters can be added using the ``add_parameter`` method.

    Parameters are exposed as atrributes of the ``Parameters``-object. This
    allows changing the values.

    Args:
        dictionary (dict): A dictionary with initial paramters and their
            associated values.
    """

    def __init__(self, dictionary=None):
        self.names = []
        if dictionary is not None:
            for key in dictionary:
                self.add_parameter(key, dictionary[key])

    def add_parameter(self, name, value):
        """Add a new parameter

        The new parameter will be available as an attribute of the object.

        Args:
            name (str): name of the new parameter
            value: value of the new parameter
        """
        if name not in self.names:
            self.names.append(name)
        setattr(self, name, value)

    def set_all(self, *args):
        """Set the value of all parameters at once.

        Args:
            args: list containing the values for all parameters

        Raises:
            ValueError: If an incorrect number of values is supplied.
        """
        if len(args) != len(self.names):
            raise ValueError("Not all parameters specified")

        for i, name in enumerate(self.names):
            setattr(self, name, args[i])

    def to_tuple(self):
        """Create a tuple of the parameter values

        Returns:
            tuple: Tuple of current parameter values in the order the
                parameters were added
        """
        return tuple(getattr(self, name) for name in self.names)

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __str__(self):
        return "Parameters " + str(
            {name: getattr(self, name)
             for name in self.names})


class ParameterTable(object):
    """A table of parameter values (for parameter scans)

    Args:
        parameters (Parameters): Initial parameter values
    """

    def __init__(self, parameters):
        self.names = parameters.names
        self.values = [[parameters[name]] for name in parameters.names]
        self.table = None
        self.filters = []
        self.constants = [i for i, _ in enumerate(self.names)]
        self.variables = []

    def set_values(self, name, values):
        """Set the value range for a certain parameter

        Args:
            name (str): name of the parameter to change
            values: iterable containing the new values for the specified
                parameter
        """
        self.values[self.names.index(name)] = [value for value in values]
        self.recalculate()

    def get_values(self, name):
        return self.values[self.names.index(name)]

    def filter_indices(self, parameters):
        filtered = copy.copy(self.table)
        for parameter in parameters:
            idx = self.names.index(parameter)
            val = parameters[parameter]
            filtered = list(filter(lambda e: (e[idx] == val), filtered))
        return [self.get_index(entry) for entry in filtered]

    def create_parameters(self, row):
        parameters = Parameters()
        for name, value in zip(self.names, row):
            parameters.add_parameter(name, value)
        return parameters

    def get_index(self, row):
        return self.table.index(row)

    def create_row_dict(self, row):
        result = {}
        for name, value in zip(self.names, row):
            result[name] = value
        return result

    def recalculate(self):
        """Recalculate all value combinations
        """
        self.table = list(itertools.product(*self.values))

        for filt in self.filters:
            new_table = list()
            for row in self.table:
                parameters = self.create_parameters(row)
                if filt(parameters):
                    new_table.append(row)
            self.table = new_table

        self.constants = [
            i for i, _ in enumerate(self.names) if len(self.values[i]) == 1
        ]
        self.variables = [
            i for i, _ in enumerate(self.names) if len(self.values[i]) > 1
        ]

    def dump(self, path):
        """Write this ``ParameterTable`` to a JSON and a pickle file

        Args:
            path (str): path to the pickle file. The ``.pickle`` extension will
                be appended if not present.
        """
        if os.path.splitext(path)[1] == ".pickle":
            path = os.path.splitext(path)[0]

        json_path = path + ".json"
        pickle_path = path + ".pickle"

        data = {
            "names": self.names,
            "values": self.values,
            "table": self.table
        }

        with open(json_path, "w") as fhandle:
            json.dump(data, fhandle)

        with open(pickle_path, "wb") as fhandle:
            pickle.dump(data, fhandle)

    def dumps(self):
        data = {
            "names": self.names,
            "values": self.values,
            "table": self.table
        }
        return pickle.dumps(data)

    def to_data_frame(self):
        return pandas.DataFrame(
            numpy.insert(
                numpy.array(self.table),
                0,
                [i for i, _ in enumerate(self.table)],
                axis=1,
            ),
            columns=["sim_index"] + self.names,
        )

    @staticmethod
    def load(path):
        """Load the ``ParameterTable`` from a pickle file

        Args:
            path (str): path of the pickle file
        """
        if os.path.splitext(path)[1] == ".pickle":
            path = os.path.splitext(path)[0]
        pickle_path = path + ".pickle"

        with open(pickle_path, "rb") as fhandle:
            data = pickle.load(fhandle)

        parameters = Parameters()
        for name in data["names"]:
            parameters.add_parameter(name, 0.)

        def pseudo_filter(parameters, initial_table):
            return parameters.to_tuple() in initial_table

        table = ParameterTable(parameters)
        table.names = data["names"]
        table.values = data["values"]
        table.table = data["table"]
        table.filters.append(
            functools.partial(pseudo_filter, initial_table=data["table"]))
        table.recalculate()

        return table

    @staticmethod
    def loads(bytes_obj):
        data = pickle.loads(bytes_obj)

        parameters = Parameters()
        for name in data["names"]:
            parameters.add_parameter(name, 0.)

        def pseudo_filter(parameters, initial_table):
            return parameters.to_tuple() in initial_table

        table = ParameterTable(parameters)
        table.names = data["names"]
        table.values = data["values"]
        table.table = data["table"]
        table.filters.append(
            functools.partial(pseudo_filter, initial_table=data["table"]))
        table.recalculate()

        return table

    def compare(self, other):
        """Compare this ParameterTable to another one

        Args:
            other (ParameterTable): The other parameter table

        Returns:
            NoneType: If the parameter tables are incompatible, ``None`` is
                returned
            dict: A dictionary that describes between the tables is returned.
                The key ``missing_rows`` yields a list of value combinations
                that are not contained in this ``ParameterTable`` compared to
                the other. ``extra_rows`` gives the value combinations that are
                only present in this table. ``moved_rows`` contains a list of
                tuples of common rows that have different indices in the two
                tables. The first number is the index in ``self``, the second
                is the index in ```other``.
        """
        if self.names != other.names:
            return None

        common_rows = set(self.table) & set(other.table)
        missing_rows = set(other.table) - set(self.table)
        extra_rows = set(self.table) - set(other.table)

        moved_rows = []
        for row in common_rows:
            i = self.table.index(row)
            j = other.table.index(row)
            if i != j:
                moved_rows.append((i, j))

        return {
            "missing_rows": missing_rows,
            "extra_rows": extra_rows,
            "moved_rows": moved_rows,
        }

    def format_table(self, fmt="plain"):
        table = [[i] + list(row) for i, row in enumerate(self.table)]
        return tabulate.tabulate(
            table, headers=["index"] + self.names, tablefmt=fmt)

    def format_constants(self, fmt="plain"):
        return tabulate.tabulate(
            [[self.names[i], self.values[i][0]] for i in self.constants],
            headers=["constant", "value"],
            tablefmt=fmt,
        )

    def format_variables(self, fmt="plain"):
        table = {self.names[i]: self.values[i] for i in self.variables}
        max_len = len(table[max(table, key=lambda x: len(table[x]))])
        for key in table:
            while len(table[key]) < max_len:
                table[key].append(None)
        return tabulate.tabulate(table, tablefmt=fmt, headers="keys")

    def __getitem__(self, e):
        if isinstance(e, str):
            return self.get_values(e)
        elif isinstance(e, tuple):
            return self.get_index(e)
        elif isinstance(e, int):
            return self.table[e]
        else:
            raise NotImplementedError(
                "ParameterTable.__getitem__ not implemented for type %s",
                str(type(e)))
