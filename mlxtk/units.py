from pathlib import Path
from typing import Any, Optional, Union

import sympy
import sympy.abc
from sympy.physics.quantum.constants import hbar

from .settings import load_settings
from .util import memoize


class Unit:
    def __init__(self, expression: Union[str, Any]):
        if isinstance(expression, str):
            self.expression = sympy.sympify(expression,
                                            locals=sympy.abc._clash)
        else:
            self.expression = sympy.simplify(expression)

    def format_label(self, quantity: str):
        return "$" + quantity + r"\:\left[" + str(self) + r"\right]$"

    def __str__(self):
        return sympy.latex(self.expression)

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit(self.expression * other.expression)
        return Unit(self.expression * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Unit(self.expression / other.expression)
        return Unit(self.expression / other)

    def __rtruediv__(self, other):
        if isinstance(other, Unit):
            return Unit(other.expression / self.expression)
        return Unit(other / self.expression)

    def __pow__(self, other):
        if isinstance(other, Unit):
            return Unit(self.expression**other.expression)
        return Unit(self.expression**other)

    def __rpow__(self, other):
        if isinstance(other, Unit):
            return Unit(other.expression**self.expression)
        return Unit(other**self.expression)


class MissingUnitError(Exception):
    def __init__(self, dimension: str):
        super().__init__("Missing unit for dimension: " + dimension)


class UnitSystem:
    def __init__(self):
        self.units = {}

    @staticmethod
    @memoize
    def from_config(working_directory: Path = Path.cwd()):
        system = UnitSystem()
        units = load_settings(working_directory).get("units", {})
        for dimension in units:
            system.units[dimension] = Unit(units[dimension])
        return system

    def get_length_unit(self) -> Unit:
        if "length" not in self.units:
            raise MissingUnitError("length")

        return self.units["length"]

    def get_time_unit(self) -> Unit:
        if "time" not in self.units:
            raise MissingUnitError("time")

        return self.units["time"]

    def get_energy_unit(self) -> Unit:
        if "energy" in self.units:
            return self.units["energy"]

        unit = Unit(hbar / self.get_time_unit().expression)
        self.units["energy"] = unit
        return unit

    def get_speed_unit(self) -> Unit:
        if "speed" in self.units:
            return self.units["speed"]

        unit = Unit(self.get_length_unit().expression /
                    self.get_time_unit().expression)
        self.units["speed"] = unit
        return unit

    def get_acceleration_unit(self) -> Unit:
        if "acceleration" in self.units:
            return self.units["acceleration"]

        unit = Unit(self.get_speed_unit().expression /
                    self.get_time_unit().expression)
        self.units["acceleration"] = unit
        return unit

    def get_delta_interaction_unit(self) -> Unit:
        if "delta_interaction" in self.units:
            return self.units["delta_interaction"]

        unit = Unit(self.get_energy_unit().expression *
                    self.get_length_unit().expression)
        self.units["delta_interaction"] = unit
        return unit

    def init_all_units(self):
        for method in dir(self):
            if not callable(getattr(self, method)):
                continue

            if not method.startswith("get_"):
                continue

            if not method.endswith("_unit"):
                continue

            try:
                getattr(self, method)()
            except MissingUnitError:
                pass

    def __str__(self):
        self.init_all_units()

        output = [
            "UnitSystem:",
        ]
        for dimension in self.units:
            output.append("\t" + dimension + ":  " +
                          str(self.units[dimension]))

        return "\n".join(output)


def get_default_unit_system(working_directory: Path = Path.cwd()
                            ) -> UnitSystem:
    return UnitSystem.from_config(working_directory)
