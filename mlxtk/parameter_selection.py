import os.path
import pickle
from typing import Any, Callable, Iterable, List, Set

from .parameters import Parameters


class ParameterSelection:
    def __init__(self, parameters: Iterable[Parameters], path: str = None):
        self.parameters = [(i, p) for i, p in enumerate(parameters)]
        self.path = path

    def fix_parameter(self, name: str, value: Any):
        return ParameterSelection(
            [entry[1] for entry in self.parameters if entry[1][name] == value],
            self.path)

    def select_parameter(self, name: str, values: Iterable[Any]):
        return ParameterSelection([
            entry[1] for entry in self.parameters if entry[1][name] in values
        ], self.path)

    def get_values(self, name: str) -> Set[Any]:
        return set((entry[1][name] for entry in self.parameters))

    def get_paths(self) -> List[str]:
        if not self.path:
            raise ValueError("No path is specified for ParameterSelection")

        return [
            os.path.join(self.path, "by_index", str(i))
            for i, _ in self.parameters
        ]

    def get_parameters(self) -> List[Parameters]:
        return [parameter for _, parameter in self.parameters]

    def foreach(self,
                func: Callable[[int, str, Parameters], Any]) -> List[Any]:
        return [
            func(entry[0], path, entry[1])
            for entry, path in zip(self.parameters, self.get_paths())
        ]

    def __str__(self):
        return "\n".join("{}: {}".format(i, p) for i, p in self.parameters)


def load_scan(path: str) -> ParameterSelection:
    with open(os.path.join(path, "scan.pickle"), "rb") as fptr:
        obj = pickle.load(fptr)
        return ParameterSelection((parameter for parameter in obj), path)
