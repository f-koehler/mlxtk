import numpy
import scipy

from .. import log
from . import gpop_diff

LOGGER = log.get_logger(__name__)


def compute_absolute_signal_diff_1d(t1, t2, sig1, sig2):
    interpolate = False

    if len(t1) != len(t2):
        LOGGER.warn("incompatible time points, interpolate")
        interpolate = True
    elif not numpy.allclose(
            t1, t2, rtol=gpop_diff.CLOSE_RTOL, atol=gpop_diff.CLOSE_ATOL):
        LOGGER.warn(
            "time points are not close to each other (rtol={}, atol={}), interpolate".
            format(gpop_diff.CLOSE_RTOL, gpop_diff.CLOSE_ATOL))
        interpolate = True

    if interpolate:
        t_min = max(min(t1), min(t2))
        t_max = min(max(t1), max(t2))
        n_t = max(len(t1), len(t2))

        t = numpy.linspace(t_min, t_max, n_t)
        interp1 = scipy.interpolate.interp1d(
            t1,
            sig1,
            kind=5,
            bounds_error=True,
            assume_sorted=True,
            copy=False)
        interp2 = scipy.interpolate.interp1d(
            t2,
            sig2,
            kind=5,
            bounds_error=True,
            assume_sorted=True,
            copy=False)
        return t, interp1(t) - interp2(t)
    else:
        return t1.copy(), sig1 - sig2


def compute_relative_signal_diff_1d(t1, t2, sig1, sig2, threshold=1e-5):
    interpolate = False

    if len(t1) != len(t2):
        LOGGER.warn("incompatible time points, interpolate")
        interpolate = True
    elif not numpy.allclose(
            t1, t2, rtol=gpop_diff.CLOSE_RTOL, atol=gpop_diff.CLOSE_ATOL):
        LOGGER.warn(
            "time points are not close to each other (rtol={}, atol={}), interpolate".
            format(gpop_diff.CLOSE_RTOL, gpop_diff.CLOSE_ATOL))
        interpolate = True

    if interpolate:
        t_min = max(min(t1), min(t2))
        t_max = min(max(t1), max(t2))
        n_t = max(len(t1), len(t2))

        t = numpy.linspace(t_min, t_max, n_t)
        s1 = scipy.interpolate.interp1d(
            t1,
            sig1,
            kind=5,
            bounds_error=True,
            assume_sorted=True,
            copy=False)(t)
        s2 = scipy.interpolate.interp1d(
            t2,
            sig2,
            kind=5,
            bounds_error=True,
            assume_sorted=True,
            copy=False)(t)
        mask = numpy.logical_or(s1 > threshold, s2 > threshold)
        values = numpy.zeros_like(sig1)
        values[mask] = 1. - s2[mask] / s1[mask]
        return t, values
    else:
        mask = numpy.logical_or(sig1 > threshold, sig2 > threshold)
        values = numpy.zeros_like(sig1)
        values[mask] = 1. - sig2[mask] / sig1[mask]
        return t1.copy(), values
