from typing import Tuple

import numpy
import scipy.interpolate
import scipy.optimize
import scipy.signal


def find_relative_maxima(x: numpy.ndarray,
                         y: numpy.ndarray,
                         order: int = 5,
                         threshold: float = None,
                         interpolation={
                             "order": 5,
                             "points": 10000
                         }) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if interpolation is not None:
        interp = scipy.interpolate.interp1d(x,
                                            y,
                                            kind=interpolation["order"],
                                            bounds_error=True)
        x = numpy.linspace(x.min(), x.max(), interpolation["points"])
        y = interp(x)

    peak_args = numpy.sort(
        numpy.array(scipy.signal.argrelmax(y, order=order)).flatten())
    if threshold is not None:
        peak_args = numpy.array(
            [arg for arg in peak_args if y[arg] >= threshold])

    if not len(peak_args):
        return numpy.array([]), numpy.array([])

    return x[peak_args], y[peak_args]


def find_relative_minima(x: numpy.ndarray,
                         y: numpy.ndarray,
                         order: int = 5,
                         interpolation={
                             "order": 5,
                             "points": 10000
                         }) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if interpolation is not None:
        interp = scipy.interpolate.interp1d(x,
                                            y,
                                            kind=interpolation["order"],
                                            bounds_error=True)
        x = numpy.linspace(x.min(), x.max(), interpolation["points"])
        y = interp(x)

    peak_args = numpy.sort(
        numpy.array(scipy.signal.argrelmin(y, order=order)).flatten())

    if not len(peak_args):
        return numpy.array([]), numpy.array([])

    return x[peak_args], y[peak_args]


def find_roots(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    interp = scipy.interpolate.interp1d(x, y, bounds_error=True, kind=5)

    signs = numpy.sign(y)
    sign_changes = numpy.argwhere(
        ((numpy.roll(signs, 1) - signs) != 0).astype(int))

    brackets = [(x[arg - 1], x[arg]) for arg in sign_changes]

    roots = []
    for bracket in brackets:
        sol = scipy.optimize.root_scalar(interp, bracket=bracket)
        if not sol.converged:
            raise RuntimeError("TODO: error message")
        roots.append(sol.root)

    return numpy.sort(numpy.array(roots))
