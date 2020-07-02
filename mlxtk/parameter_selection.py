"""Selecting simulations from a scan based on parameters.
"""
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy
from tqdm import tqdm

from mlxtk.cwd import WorkingDir
from mlxtk.log import get_logger
from mlxtk.parameters import Parameters, get_variables
from mlxtk.util import make_path, map_parallel_progress

LOGGER = get_logger(__name__)


class ParameterSelection:
    def __init__(
        self,
        parameters: Iterable[Parameters],
        path: Union[str, Path] = None,
        indices: Iterable[int] = None,
    ):
        if indices is None:
            self.parameters = [(i, p) for i, p in enumerate(parameters)]
        else:
            self.parameters = list(zip(indices, parameters))
        self.path = None if path is None else make_path(path).resolve()

    def get_variable_names(self) -> List[str]:
        return get_variables([p[1] for p in self.parameters])[0]

    def copy(self):
        return ParameterSelection(
            deepcopy(
                [p[1].copy() for p in self.parameters],
                self.path,
                [p[0] for p in self.parameters],
            )
        )

    def partition_single(self, parameter_name: str):
        parameter_values = self.get_values(parameter_name)
        partitions = {
            value: [[], []] for value in parameter_values
        }  # type: Dict[Any, List[List[int], List[Parameters]]]
        for index, parameter in self.parameters:
            partitions[parameter[parameter_name]][0].append(index)
            partitions[parameter[parameter_name]][1].append(
                parameter.copy().remove_parameter(parameter_name)
            )
        return {
            value: ParameterSelection(
                partitions[value][1], self.path, partitions[value][0]
            )
            for value in parameter_values
        }

    def partition(self, parameter_names: Union[str, List[str]]):
        if isinstance(parameter_names, str):
            return self.partition_single(parameter_names)
        raise NotImplementedError(
            "Only partitioning according to a single parameter is implemented yet."
        )

    def fix_parameter(self, name: str, value: Any) -> "ParameterSelection":
        """Select by the value of a single parameter.

        Args:
            name: Name of the parameter to fix.
            value: Desired value for the parameter.

        Returns:
            A new ParameterSelection containing only matching parameter sets.
        """
        entries = []
        indices = []
        for index, entry in self.parameters:
            if entry[name] == value:
                entries.append(entry)
                indices.append(index)
        return ParameterSelection(entries, self.path, indices)

    def group_by(self, name: str):
        return {
            value: self.fix_parameter(name, value) for value in self.get_values(name)
        }

    def select_parameter(self, name: str, values: Iterable[Any]):
        """Select by multiple values of a single parameter.

        Args:
            name: Name of the parameter to fix.
            value: Desired values for the parameter.

        Returns:
            A new ParameterSelection containing only matching parameter sets.
        """
        return ParameterSelection(
            [entry[1] for entry in self.parameters if entry[1][name] in values],
            self.path,
            [entry[0] for entry in self.parameters if entry[1][name] in values],
        )

    def select_parameters(self, names: Iterable[str], values: Iterable[Iterable[Any]]):
        """Select by multiple values of a single parameter.

        Args:
            name: Name of the parameter to fix.
            value: Desired values for the parameter.

        Returns:
            A new ParameterSelection containing only matching parameter sets.
        """
        selection = self
        for name, vals in zip(names, values):
            selection = selection.select_parameter(name, vals)
        return selection

    def get_values(self, name: str) -> Set[Any]:
        """Get all unique values for a parameter.

        Args:
            name: Name of the parameter.

        Returns:
            All unique values of the given parameter.
        """
        return list(set((entry[1][name] for entry in self.parameters)))

    def get_path(self, parameters: Parameters) -> Path:
        for entry, path in zip(self.parameters, self.get_paths()):
            if parameters.has_same_common_parameters(entry[1]):
                return path

        raise RuntimeError("cannot find path for parameters: " + str(parameters))

    def get_paths(self) -> List[Path]:
        """Compute the paths for all included parameter sets.

        Raises:
            ValueError: No path is provided for this ParameterSelection

        Returns:
            Paths of all included parameter sets.
        """
        if not self.path:
            raise ValueError("No path is specified for ParameterSelection")

        return [self.path / "by_index" / str(i) for i, _ in self.parameters]

    def generate_paths(self, subpath: Union[str, Path]) -> List[Path]:
        return [Path(path) / subpath for path in self.get_paths()]

    def get_parameters(self) -> List[Parameters]:
        """Get all included parameters sets.

        Returns:
            A list of all included parameter sets.
        """
        return [parameter for _, parameter in self.parameters]

    def get_variable_values(self) -> Tuple[List[str], Dict[str, numpy.array]]:
        variables = self.get_variable_names()
        values = {var: [] for var in variables}

        for _, parameters in self.parameters:
            for var in variables:
                values[var].append(parameters[var])

        for var in variables:
            values[var] = numpy.array(values[var])

        return variables, values

    def foreach(
        self, func: Callable[[int, str, Parameters], Any], parallel=True
    ) -> List[Any]:
        """Call a function for each included parameter set.

        Args:
            func: Function to call for each parameter set. It takes the index
                of the parameter set as the first argument, the path as a
                second argument and the parameter set as the third argument.

        Returns:
            The provided function may return values. This function returns a
            list of all return values created by calling the function for each
            parameter set.
        """

        def helper(item):
            return func(item[0], item[1], item[2])

        work = [
            [entry[0], path, entry[1]]
            for entry, path in zip(self.parameters, self.get_paths())
        ]

        if parallel:
            return map_parallel_progress(helper, work)

        return [
            func(entry[0], path, entry[1])
            for entry, path in tqdm(list(zip(self.parameters, self.get_paths())))
        ]

    def plot_foreach(
        self, name: str, func: Callable[[int, str, Parameters], None]
    ) -> Optional[List[Any]]:
        if not self.path:
            raise RuntimeError("No path set for parameter selection")

        plot_dir = self.path / "plots" / name
        if not plot_dir.exists():
            os.makedirs(plot_dir)

        with WorkingDir(plot_dir):
            return self.foreach(func)

    def __str__(self):
        return "\n".join("{}: {}".format(i, p) for i, p in self.parameters)


def load_scan(path: Union[str, Path]) -> ParameterSelection:
    """Load all parameter sets of a parameter scan.

    Args:
        path: Path to the parameter scan containing the file ``scan.pickle``.
    """
    path = make_path(path)
    with open(path / "scan.pickle", "rb") as fptr:
        obj = pickle.load(fptr)
        return ParameterSelection((parameter for parameter in obj), path)


def group_scans_by(
    selections: List[ParameterSelection], parameter_name: str
) -> Dict[Any, List[ParameterSelection]]:
    values = set()
    for selection in selections:
        scan_values = selection.get_values(parameter_name)
        if len(scan_values) != 1:
            raise RuntimeError(
                'exactly one parameter for "{}" required per scan'.format(
                    parameter_name
                )
            )
        values.add(scan_values.pop())

    result = {}
    for selection in selections:
        val = selection.parameters[0][1][parameter_name]
        if parameter_name not in result:
            result[parameter_name] = []
        result[parameter_name].append(val)

    return result
