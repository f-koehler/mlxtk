import numpy
import scipy

from .. import log

LOGGER = log.get_logger(__name__)
CLOSE_RTOL = 1e-5
CLOSE_ATOL = 1e-8


def compute_absolute_gpop_diff(time1, time2, grid1, grid2, gpop1, gpop2):
    """Compute the absolute difference between one-body densities :math:`\\rho_1(x,t)-\\rho_2(x,t)`

    When incompatible grids/times are specified, interpolation routines are applied.

    Args:
        time1: time points of the first density
        time2: time points of the second density
        grid1: grid points of the first density
        grid2: grid points of the second density
        gpop1: first density
        gpop2: second density

    Returns:
        Time points, grid points, absolute difference between the densities
    """
    interpolate = False

    if len(time1) != len(time2):
        LOGGER.warn("different number of time points, interpolate")
        interpolate = True
    elif len(grid1) != len(grid2):
        LOGGER.warn("different number of grid points, interpolate")
        interpolate = True
    elif not numpy.allclose(time1, time2):
        LOGGER.warn(
            "time points are not close to each other (rtol={}, atol={}), interpolate".
            format(CLOSE_RTOL, CLOSE_ATOL))
        interpolate = True
    elif not numpy.allclose(grid1, grid2):
        LOGGER.warn(
            "grid points are not close to each other (rtol={}, atol={}), interpolate".
            format(CLOSE_RTOL, CLOSE_ATOL))
        interpolate = True

    if interpolate:
        den1 = numpy.transpose(gpop1)
        den2 = numpy.transpose(gpop2)

        interp1 = scipy.interpolate.interp2d(
            time1, grid1, den1, kind=5, copy=False, bounds_error=True)
        interp2 = scipy.interpolate.interp2d(
            time2, grid2, den2, kind=5, copy=False, bounds_error=True)

        t_min = max(time1.min(), time2.min())
        t_max = min(time1.max(), time2.max())
        x_min = max(grid1.min(), grid2.min())
        x_max = min(grid1.max(), grid2.max())
        n_t = max(len(time1), len(time2))
        n_x = max(len(grid1), len(grid2))

        t = numpy.linspace(t_min, t_max, n_t)
        x = numpy.linspace(x_min, x_max, n_x)

        return t, x, numpy.transpose(interp1(t, x) - interp2(t, x))
    else:
        return time1.copy(), grid1.copy(), gpop1 - gpop2


def compute_relative_gpop_diff(time1,
                               time2,
                               grid1,
                               grid2,
                               gpop1,
                               gpop2,
                               threshold=1e-3):
    """Compute the relative difference between one-body densities :math:`1-\\frac{\\rho_2(x,t)}{\\rho_1(x,t)}`

    Only points where :math:`\\rho_1(x,t)` or :math:`\\rho_2(x,t)` excels the threshold are taken into account to avoid divergences.
    When incompatible grids/times are specified, interpolation routines are applied.

    Args:
        time1: time points of the first density
        time2: time points of the second density
        grid1: grid points of the first density
        grid2: grid points of the second density
        gpop1: first density
        gpop2: second density
        threshold:

    Returns:
        Time points, grid points, relative difference between the densities
    """
    interpolate = False

    if len(time1) != len(time2):
        LOGGER.warn("different number of time points, interpolate")
        interpolate = True
    elif len(grid1) != len(grid2):
        LOGGER.warn("different number of grid points, interpolate")
        interpolate = True
    elif not numpy.allclose(time1, time2):
        LOGGER.warn(
            "time points are not close to each other (rtol={}, atol={}), interpolate".
            format(CLOSE_RTOL, CLOSE_ATOL))
        interpolate = True
    elif not numpy.allclose(grid1, grid2):
        LOGGER.warn(
            "grid points are not close to each other (rtol={}, atol={}), interpolate".
            format(CLOSE_RTOL, CLOSE_ATOL))
        interpolate = True

    if interpolate:
        t_min = max(time1.min(), time2.min())
        t_max = min(time1.max(), time2.max())
        x_min = max(grid1.min(), grid2.min())
        x_max = min(grid1.max(), grid2.max())
        n_t = max(len(time1), len(time2))
        n_x = max(len(grid1), len(grid2))

        t = numpy.linspace(t_min, t_max, n_t)
        x = numpy.linspace(x_min, x_max, n_x)

        den1 = scipy.interpolate.interp2d(
            time1,
            grid1,
            numpy.transpose(gpop1),
            kind=5,
            copy=False,
            bounds_error=True)(t, x)
        den2 = scipy.interpolate.interp2d(
            time2,
            grid2,
            numpy.transpose(gpop2),
            kind=5,
            copy=False,
            bounds_error=True)(t, x)

        mask = numpy.logical_and(den1 > threshold, den2 > threshold)
        values = numpy.zeros_like(den1)
        values[mask] = 1. - den2[mask] / den1[mask]
        return t, x, values
    else:
        mask = numpy.logical_and(gpop1 > threshold, gpop2 > threshold)
        values = numpy.zeros_like(gpop1)
        # values[mask] = 1. - gpop2[mask] / gpop1[mask]
        values[mask] = (gpop1[mask] - gpop2[mask]) / (
            numpy.abs(gpop1[mask]) + numpy.abs(gpop2[mask]))
        return time1.copy(), grid1.copy(), values


def compute_integrated_absolute_gpop_diff(time, grid, weights, gpop1, gpop2):
    _, _, diffs = compute_absolute_gpop_diff(time, time, grid, grid, gpop1,
                                             gpop2)
    return time.copy(), numpy.sum(numpy.abs(diffs * weights), axis=1) / 2.


def compute_integrated_relative_gpop_diff(time,
                                          grid,
                                          weights,
                                          gpop1,
                                          gpop2,
                                          threshold=1e-3):
    # _, _, diffs = compute_relative_gpop_diff(time, time, grid, grid, gpop1,
    #                                          gpop2, threshold)
    # return time.copy(), numpy.sum(diffs, axis=1)
    mask = numpy.logical_and(gpop1 > threshold, gpop2 > threshold)
    diff = numpy.zeros_like(gpop1)
    # diff[mask] = numpy.abs(gpop1[mask] - gpop2[mask]) / (
    #     numpy.abs(gpop1[mask]) + numpy.abs(gpop2[mask]))
    diff[mask] = numpy.abs(1. - gpop2[mask] / gpop1[mask])
    return time.copy(), numpy.sum(weights * diff, axis=1)
