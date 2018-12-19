import copy
import itertools
import json
from typing import (Any, Callable, Dict, Generator, Iterable, List, Optional,
                    Tuple, Union)


class Parameters:
    """A class to hold arbitrary simulation parameters

    This class helps to merge all simulation parameters into one variable and
    to document them.
    All parameters are exposed as member variables of this class.
    """

    def __init__(self, params: List[Dict[str, Any]]=[]):
        self.names = []  # type: List[str]
        self.docs = {}  # type: Dict[str, str]

        for param in params:
            Parameters.__iadd__(self, param)

    def add_parameter(self, name: str, value: Optional[Any]=None, doc: str=""):
        """Add a new parameter

        Args:
            name (str): name of the parameter, should also be a valid python
                variable name
            value: value for the parameter
            doc (str): description of the purpose of this parameter
        """
        self.names.append(name)
        self.docs[name] = doc
        self.__setitem__(name, value)

    def to_json(self) -> str:
        return json.dumps({
            "values": {name: self[name]
                       for name in self.names},
            "docs": {name: self.docs[name]
                     for name in self.names},
        })

    def set_values(self, values: Iterable[Any]):
        for name, value in zip(self.names, values):
            self.__setitem__(name, value)

    def copy(self):
        p = Parameters()
        for name in self.names:
            p.add_parameter(name,
                            copy.deepcopy(self.__getitem__(name)),
                            self.docs[name])
        return p

    def __iadd__(self, param: Union[dict, list]):
        if isinstance(param, dict):
            self.add_parameter(**param)
        else:
            self.add_parameter(*param)
        return self

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "values": {name: self.__getitem__(name)
                       for name in self.names},
            "docs": self.docs,
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.names = []
        self.docs = {}
        for name in state["values"]:
            self.add_parameter(name, state["values"][name], state["docs"].get(
                name, ""))

    def __repr__(self) -> str:
        return "_".join([name + "=" + str(self[name]) for name in self.names])

    def __str__(self) -> str:
        return ("{\n" + "\n".join([
            "  {}:\n    value: {}\n    doc:   {}".format(
                name, self.__getitem__(name), self.docs[name])
            for name in self.names
        ]) + "\n}")

    def __eq__(self, other) -> bool:
        if isinstance(other, Parameters):
            if self.names != other.names:
                return False

            for name in self.names:
                if self.__getitem__(name) != other.__getitem__(name):
                    return False

            return True

        raise NotImplementedError

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)


def generate_all(parameters: Parameters,
                 values: Dict[str, Any]) -> Generator[Parameters, None, None]:
    for name in parameters.names:
        values[name] = values.get(name, [parameters[name]])

    for combination in itertools.product(
            * [values[name] for name in parameters.names]):
        p = copy.deepcopy(parameters)
        p.set_values(combination)
        yield p


def select(
        combinations: Generator[Parameters, None, None],
        condition: Callable[[Parameters], bool],
) -> Generator[Parameters, None, None]:
    for combination in combinations:
        if condition(combination):
            yield combination


def add(combinations: Generator[Parameters, None, None],
        new_combination: Parameters) -> Generator[Parameters, None, None]:
    for combination in combinations:
        yield combination
    yield new_combination


def add_multiple(
        combinations: Generator[Parameters, None, None],
        new_combinations: Iterable[Parameters],
) -> Generator[Parameters, None, None]:
    for combination in combinations:
        yield combination

    for combination in new_combinations:
        yield combination


def merge(
        combinations1: Generator[Parameters, None, None],
        combinations2: Generator[Parameters, None, None],
) -> Generator[Parameters, None, None]:
    for combination in combinations1:
        yield combination

    for combination in combinations2:
        yield combination


def get_variables(parameters: List[Parameters]) -> Tuple[List[str], List[str]]:
    p0 = parameters[0]
    is_variable = [False for n in p0.names]

    for p in parameters[1:]:
        for i, n in enumerate(p.names):
            if is_variable[i]:
                continue

            if p[n] != p0[n]:
                is_variable[i] = True

    variables = []
    constants = []
    for name, value in zip(p0.names, is_variable):
        if value:
            variables.append(name)
        else:
            constants.append(name)

    return variables, constants
