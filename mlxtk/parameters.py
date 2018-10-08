import copy
import itertools
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Set, Union


class Parameters(object):
    """A class to hold arbitrary simulation parameters

    This class helps to merge all simulation parameters into one variable and
    to document them.
    All parameters are exposed as member variables of this class.
    """

    def __init__(self, params: Set = set()):
        self.names = set()  # type: Set[str]
        self.docs = {}  # type: Dict[str, str]

        for param in params:
            Parameters.__iadd__(self, param)

    def add_parameter(self, name: str, value: Optional[Any] = None, doc: str = ""):
        """Add a new parameter

        Args:
            name (str): name of the parameter, should also be a valid python variable name
            value: value for the parameter
            doc (str): description of the purpose of this parameter
        """
        self.names.add(name)
        self.docs[name] = doc
        setattr(self, name, value)

    def set_values(self, values: Iterable[Any]):
        for name, value in zip(self.names, values):
            setattr(self, name, value)

    def __iadd__(self, param: Union[dict, list]):
        if isinstance(param, dict):
            self.add_parameter(**param)
        else:
            self.add_parameter(*param)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "values": [getattr(self, name) for name in self.names],
            "docs": self.docs,
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.names = set()
        self.docs = {}
        for entry in state:
            self.add_parameter(**state[entry])


def generate_all(
    parameters: Parameters, values: Dict[str, Any]
) -> Generator[Parameters, None, None]:
    for name in parameters.names:
        values[name] = values.get(name, [getattr(parameters, name)])

    for combination in itertools.product(*[values[name] for name in parameters.names]):
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


def add(
    combinations: Generator[Parameters, None, None], new_combination: Parameters
) -> Generator[Parameters, None, None]:
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
