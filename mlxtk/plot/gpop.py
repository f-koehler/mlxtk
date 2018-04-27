import numpy
import scipy

from ..inout.gpop import read_gpop
from .. import log


def plot_gpop(plot, path, dof):
    times, grids, densities = read_gpop(path)
    density = numpy.transpose(densities[dof])
    t, x = numpy.meshgrid(times, grids[dof])
    heatmap = plot.axes.pcolormesh(t, x, density, cmap="gnuplot")
    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel("$x$")
    cbar = plot.figure.colorbar(heatmap)
    cbar.ax.set_ylabel(r"$\rho_1(x)$")


def plot_gpop_diff(plot, path1, path2, dof, relative=False):
    logger = log.get_logger(__name__)

    threshold = 1e-5

    times1, grids1, densities1 = read_gpop(path1)
    times2, grids2, densities2 = read_gpop(path2)

    grid1 = grids1[dof]
    grid2 = grids2[dof]

    interpolate = False
    if not numpy.array_equal(times1, times2):
        if not numpy.array_equal(grids1[dof], grids2[dof]):
            logger.warn("incompatible time and grid points, interpolating")
        else:
            logger.warn("incompatible time points, interpolating")
        interpolate = True
    elif not numpy.array_equal(grids1[dof], grids2[dof]):
        logger.warn("incompatible grid points, interpolating")
        interpolate = True

    if not interpolate:
        if relative:
            mask = numpy.logical_or(densities1[dof] > threshold,
                                    densities2[dof] > threshold)
            values = numpy.zeros_like(densities1[dof])
            values[mask] = 1. - densities2[dof][mask] / densities1[dof][mask]
        else:
            values = densities1[dof] - densities2[dof]
        values = values.transpose()
        amplitude = max(abs(numpy.min(values)), numpy.max(values))
        t, x = numpy.meshgrid(times1, grids1[dof])
        heatmap = plot.axes.pcolormesh(
            t, x, values, cmap="bwr", vmin=-amplitude, vmax=amplitude)
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel("$x$")
        cbar = plot.figure.colorbar(heatmap)
        if relative:
            cbar.ax.set_ylabel(
                r"$1-\frac{\rho_1^{(2)}(x,t)}{\rho_1^{(1)}(x,t)}$")
        else:
            cbar.ax.set_ylabel(r"$\rho_1^{(1)}(x,t)-\rho_1^{(2)}(x,t)$")
    else:
        t_min = max(numpy.min(times1), numpy.min(times2))
        t_max = min(numpy.max(times1), numpy.max(times2))
        x_min = max(numpy.min(grid1), numpy.min(grid2))
        x_max = min(numpy.max(grid1), numpy.max(grid2))
        n_t = max(len(times1), len(times2))
        n_x = max(len(grid1), len(grid2))

        gpop1 = numpy.transpose(densities1[dof])
        gpop2 = numpy.transpose(densities2[dof])

        interp1 = scipy.interpolate.interp2d(
            times1,
            grid1,
            gpop1,
            kind="quintic",
            copy=False,
            bounds_error=True)
        interp2 = scipy.interpolate.interp2d(
            times2,
            grid2,
            gpop2,
            kind="quintic",
            copy=False,
            bounds_error=True)

        t = numpy.linspace(t_min, t_max, n_t)
        x = numpy.linspace(x_min, x_max, n_x)
        if relative:
            g1 = interp1(t, x)
            g2 = interp2(t, x)
            mask = numpy.logical_or(g1 > threshold, g2 > threshold)
            values = numpy.zeros_like(g1)
            values[mask] = (1. - g2 / g1)[mask]
        else:
            values = interp1(t, x) - interp2(t, x)
        values = values.transpose()
        amplitude = max(abs(numpy.min(values)), numpy.max(values))

        heatmap = plot.axes.pcolormesh(
            t, x, values, cmap="bwr", vmin=-amplitude, vmax=amplitude)
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel("$x$")
        cbar = plot.figure.colorbar(heatmap)
        if relative:
            cbar.ax.set_ylabel(
                r"$1-\frac{\rho_1^{(2)}(x,t)}{\rho_1^{(1)}(x,t)}$ (interpolated)"
            )
        else:
            cbar.ax.set_ylabel(
                r"$\rho_1^{(1)}(x,t)-\rho_1^{(2)}(x,t)$ (interpolated)")
