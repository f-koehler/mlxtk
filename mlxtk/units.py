from pathlib import Path
from typing import Any, Optional, Union

import sympy
import sympy.abc
from sympy.physics.quantum.constants import hbar

from mlxtk.settings import load_settings
from mlxtk.util import memoize


class Unit:
    def __init__(self, expression: Union[str, Any]):
        if isinstance(expression, str):
            self.expression = sympy.sympify(expression, locals=sympy.abc._clash)
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
            return Unit(self.expression ** other.expression)
        return Unit(self.expression ** other)

    def __rpow__(self, other):
        if isinstance(other, Unit):
            return Unit(other.expression ** self.expression)
        return Unit(other ** self.expression)

    def is_arbitrary(self) -> bool:
        return False


class ArbitraryUnit:
    def __init__(self):
        pass

    def format_label(self, quantity: str):
        return "$" + quantity + r"\:\left[\mathrm{a.u.}\right]$"

    def is_arbitrary(self) -> bool:
        return True

    def __str__(self):
        return r"\mathrm{a.u.}"

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __pow__(self, other):
        return self

    def __rpow__(self, other):
        return self


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

    def get_length_unit(self) -> Union[Unit, ArbitraryUnit]:
        if "length" not in self.units:
            return ArbitraryUnit()

        return self.units["length"]

    def get_time_unit(self) -> Union[Unit, ArbitraryUnit]:
        if "time" not in self.units:
            return ArbitraryUnit()

        return self.units["time"]

    def get_energy_unit(self) -> Union[Unit, ArbitraryUnit]:
        if "energy" in self.units:
            return self.units["energy"]

        time_unit = self.get_time_unit()
        if time_unit.is_arbitrary():
            return ArbitraryUnit()

        unit = Unit(hbar / time_unit.expression)
        self.units["energy"] = unit
        return unit

    def get_speed_unit(self) -> Union[Unit, ArbitraryUnit]:
        if "speed" in self.units:
            return self.units["speed"]

        length_unit = self.get_length_unit()
        time_unit = self.get_time_unit()
        if length_unit.is_arbitrary() or time_unit.is_arbitrary():
            return ArbitraryUnit()

        unit = Unit(length_unit.expression / time_unit.expression)
        self.units["speed"] = unit
        return unit

    def get_acceleration_unit(self) -> Union[Unit, ArbitraryUnit]:
        if "acceleration" in self.units:
            return self.units["acceleration"]

        speed_unit = self.get_speed_unit()
        time_unit = self.get_time_unit()
        if speed_unit.is_arbitrary() or time_unit.is_arbitrary():
            return ArbitraryUnit()

        unit = Unit(speed_unit.expression / time_unit.expression)
        self.units["acceleration"] = unit
        return unit

    def get_delta_interaction_unit(self) -> Union[Unit, ArbitraryUnit]:
        if "delta_interaction" in self.units:
            return self.units["delta_interaction"]

        energy_unit = self.get_energy_unit()
        length_unit = self.get_length_unit()
        if energy_unit.is_arbitrary() or length_unit.is_arbitrary():
            return ArbitraryUnit()
        unit = Unit(energy_unit.expression * length_unit.expression)
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
            output.append("\t" + dimension + ":  " + str(self.units[dimension]))

        return "\n".join(output)


def get_default_unit_system(working_directory: Path = Path.cwd()) -> UnitSystem:
    return UnitSystem.from_config(working_directory)
