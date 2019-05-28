from pathlib import Path
from typing import Any, Optional, Union

import sympy
import sympy.abc

from .settings import load_settings


def load_unit(quantity: str,
              working_directory: Path = Path.cwd()) -> Optional[Any]:
    unit = load_settings(working_directory).get("units",
                                                {}).get(quantity, None)
    if unit:
        return sympy.sympify(unit, locals=sympy.abc._clash)
    return None


def simplify(expr):
    return sympy.simplify(expr)


def latex(expr):
    return sympy.latex(expr)


def format_label(quantity: str, unit) -> str:
    return "$" + quantity + r"\:\left[" + unit + r"\right]$"


def get_length_label(quantity: str = "x",
                     working_directory: Path = Path.cwd()) -> str:
    unit = load_unit("length", working_directory)
    if unit:
        return format_label(quantity, latex(simplify(unit)))
    return format_label(quantity, r"\mathrm{a.u.}")


def get_time_label(quantity: str = "t",
                   working_directory: Path = Path.cwd()) -> str:
    unit = load_unit("time", working_directory)
    if unit:
        return format_label(quantity, latex(simplify(unit)))
    return format_label(quantity, r"\mathrm{a.u.}")


def get_energy_label(quantity: str = "E",
                     working_directory: Path = Path.cwd()) -> str:
    unit = load_unit("energy", working_directory)
    if unit:
        return format_label(quantity, latex(simplify(unit)))
    return format_label(quantity, r"\mathrm{a.u.}")
