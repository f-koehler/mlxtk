"""Selecting simulations from a scan based on parameters.
"""
import os.path
import pickle
from typing import Any, Callable, Iterable, List, Optional, Set

from .cwd import WorkingDir
from .log import redirect_for_tqdm, tqdm
from .parameters import Parameters


class ParameterSelection:
    def __init__(self, parameters: Iterable[Parameters], path: str = None):
        self.parameters = [(i, p) for i, p in enumerate(parameters)]
        self.path = os.path.abspath(path)

    def fix_parameter(self, name: str, value: Any):
        """Select by the value of a single parameter.

        Args:
            name: Name of the parameter to fix.
            value: Desired value for the parameter.

        Returns:
            A new ParameterSelection containing only matching parameter sets.
        """
        return ParameterSelection(
            [entry[1] for entry in self.parameters if entry[1][name] == value],
            self.path)

    def select_parameter(self, name: str, values: Iterable[Any]):
        """Select by multiple values of a single parameter.

        Args:
            name: Name of the parameter to fix.
            value: Desired values for the parameter.

        Returns:
            A new ParameterSelection containing only matching parameter sets.
        """
        return ParameterSelection([
            entry[1] for entry in self.parameters if entry[1][name] in values
        ], self.path)

    def get_values(self, name: str) -> Set[Any]:
        """Get all unique values for a parameter.

        Args:
            name: Name of the parameter.

        Returns:
            All unique values of the given parameter.
        """
        return list(set((entry[1][name] for entry in self.parameters)))

    def get_paths(self) -> List[str]:
        """Compute the paths for all included parameter sets.

        Raises:
            ValueError: No path is provided for this ParameterSelection

        Returns:
            Paths of all included parameter sets.
        """
        if not self.path:
            raise ValueError("No path is specified for ParameterSelection")

        return [
            os.path.join(self.path, "by_index", str(i))
            for i, _ in self.parameters
        ]

    def get_parameters(self) -> List[Parameters]:
        """Get all included parameters sets.

        Returns:
            A list of all included parameter sets.
        """
        return [parameter for _, parameter in self.parameters]

    def foreach(self,
                func: Callable[[int, str, Parameters], Any]) -> List[Any]:
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
        with redirect_for_tqdm() as original:
            results = []
            for entry, path in tqdm(
                    list(zip(self.parameters, self.get_paths())),
                    file=original,
                    dynamic_ncols=True):
                results.append(func(entry[0], path, entry[1]))

            return results

    def plot_foreach(self, name: str,
                     func: Callable[[int, str, Parameters], None]
                     ) -> Optional[List[Any]]:
        if not self.path:
            raise RuntimeError("No path set for parameter selection")

        plot_dir = os.path.join(self.path, "plots", name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        with WorkingDir(plot_dir):
            return self.foreach(func)

    def __str__(self):
        return "\n".join("{}: {}".format(i, p) for i, p in self.parameters)


def load_scan(path: str) -> ParameterSelection:
    """Load all parameter sets of a parameter scan.

    Args:
        path: Path to the parameter scan containing the file ``scan.pickle``.
    """
    with open(os.path.join(path, "scan.pickle"), "rb") as fptr:
        obj = pickle.load(fptr)
        return ParameterSelection((parameter for parameter in obj), path)
