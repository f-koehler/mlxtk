from typing import Tuple

import numpy
import scipy.fftpack
import scipy.interpolate
import scipy.optimize
import scipy.signal


def find_relative_maxima(
    x: numpy.ndarray,
    y: numpy.ndarray,
    order: int = 5,
    threshold: float = None,
    interpolation={"order": 5, "points": 10000},
    sort: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if interpolation is not None:
        interp = scipy.interpolate.interp1d(
            x, y, kind=interpolation["order"], bounds_error=True
        )
        x = numpy.linspace(x.min(), x.max(), interpolation["points"])
        y = interp(x)

    peak_args = numpy.sort(
        numpy.array(scipy.signal.argrelmax(y, order=order)).flatten()
    )
    if threshold is not None:
        peak_args = numpy.array([arg for arg in peak_args if y[arg] >= threshold])

    if not len(peak_args):
        return numpy.array([]), numpy.array([])

    x_peaks = x[peak_args]
    y_peaks = y[peak_args]

    if sort:
        permutation = numpy.argsort(y_peaks)
        x_peaks = x_peaks[permutation][::-1]
        y_peaks = y_peaks[permutation][::-1]

    return x_peaks, y_peaks


def find_relative_minima(
    x: numpy.ndarray,
    y: numpy.ndarray,
    order: int = 5,
    interpolation={"order": 5, "points": 10000},
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if interpolation is not None:
        interp = scipy.interpolate.interp1d(
            x, y, kind=interpolation["order"], bounds_error=True
        )
        x = numpy.linspace(x.min(), x.max(), interpolation["points"])
        y = interp(x)

    peak_args = numpy.sort(
        numpy.array(scipy.signal.argrelmin(y, order=order)).flatten()
    )

    if not len(peak_args):
        return numpy.array([]), numpy.array([])

    return x[peak_args], y[peak_args]


def find_roots(x: numpy.ndarray, y: numpy.ndarray, min_spacing=1e-8) -> numpy.ndarray:
    interp = scipy.interpolate.interp1d(x, y, bounds_error=True, kind=5)

    signs = numpy.sign(y)
    sign_changes = numpy.argwhere(((numpy.roll(signs, 1) - signs) != 0).astype(int))

    brackets = [(x[arg - 1], x[arg]) for arg in sign_changes]

    roots = []
    for bracket in brackets:
        sol = scipy.optimize.root_scalar(interp, bracket=bracket)
        if not sol.converged:
            raise RuntimeError("TODO: error message")
        roots.append(sol.root)

    roots = numpy.sort(numpy.array(roots))
    filtered_roots = [roots[0]]
    for i in range(len(roots) - 1):
        if roots[i + 1] - roots[i] > min_spacing:
            filtered_roots.append(roots[i + 1])
    return numpy.array(filtered_roots)


def find_sign_changes(x: numpy.ndarray) -> numpy.ndarray:
    signs = numpy.sign(x)
    return numpy.roll(signs, 1) - signs != 0


def fourier_transform(
    t: numpy.ndarray, signal: numpy.ndarray, only_positive: bool = True
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    amplitudes = numpy.abs(scipy.fftpack.fftshift(scipy.fftpack.fft(signal)))
    amplitudes = amplitudes / amplitudes.max()
    frequencies = scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(t), t[1] - t[0]))
    if only_positive:
        amplitudes = amplitudes[frequencies >= 0]
        frequencies = frequencies[frequencies >= 0]
    return frequencies, amplitudes
