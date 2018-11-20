import numpy
import scipy

from ..inout.gpop import read_gpop
from ..tools.gpop_diff import compute_absolute_gpop_diff, compute_relative_gpop_diff, compute_integrated_relative_gpop_diff, compute_integrated_absolute_gpop_diff
from .. import log


def plot_gpop(plot, path, dof, **kwargs):
    times, grids, densities = read_gpop(path)
    density = numpy.transpose(densities[dof])
    t, x = numpy.meshgrid(times, grids[dof])
    heatmap = plot.axes.pcolormesh(t, x, density, cmap="gnuplot", shading="gouraud" if kwargs.get("smooth", False) else "flat")
    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel("$x$")

    cbar = plot.figure.colorbar(heatmap)
    cbar.ax.set_ylabel(r"$\rho_1(x)$")


def plot_gpop_diff(plot, path1, path2, dof, relative=False, threshold=1e-5):
    """
    .. todo:: allow vmin != -vmax but still keep the white color for a value of 0
    """
    times1, grids1, densities1 = read_gpop(path1)
    times2, grids2, densities2 = read_gpop(path2)

    grid1 = grids1[dof]
    grid2 = grids2[dof]

    if relative:
        label = r"$1-\frac{\rho_1^{(2)}(x,t)}{\rho_1^{(1)}(x,t)}$"
        t, x, values = compute_relative_gpop_diff(
            times1,
            times2,
            grid1,
            grid2,
            densities1[dof],
            densities2[dof],
            threshold=threshold,
        )
    else:
        label = r"$\rho_1^{(1)}(x,t)-\rho_1^{(2)}(x,t)$"
        t, x, values = compute_absolute_gpop_diff(
            times1, times2, grid1, grid2, densities1[dof], densities2[dof])

    amplitude = max(abs(values.min()), abs(values.max()))

    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel("$x$")
    T, X = numpy.meshgrid(t, x)
    heatmap = plot.axes.pcolormesh(
        T, X, values.transpose(), cmap="bwr", vmin=-amplitude, vmax=amplitude)
    cbar = plot.figure.colorbar(heatmap)
    cbar.ax.set_ylabel(label)


def plot_integrated_gpop_diff(plot,
                              path1,
                              path2,
                              dof,
                              relative=False,
                              threshold=1e-5):
    times, grids, densities1 = read_gpop(path1)
    _, _, densities2 = read_gpop(path2)

    grid = grids[dof]

    if relative:
        label = r"$\int\left(1-\frac{\rho^{(2)}(x,t)}{\rho^{(1)}(x,t)}\right)\mathrm{d}x$"
        t, d = compute_integrated_relative_gpop_diff(
            times, grid, densities1[dof], densities2[dof], threshold)
    else:
        label = r"$\frac{1}{2}\int\left|\rho^{(1)}(x,t)-\rho^{(2)}(x,t)\right|\mathrm{d}x$"
        t, d = compute_integrated_absolute_gpop_diff(
            times, grid, densities1[dof], densities2[dof])

    plot.axes.set_xlabel("$t$")
    plot.axes.set_ylabel(label)
    plot.axes.plot(t, d)
