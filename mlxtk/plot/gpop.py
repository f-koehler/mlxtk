from typing import Optional

import matplotlib.colors
import matplotlib.pyplot
import matplotlib.tri
import numpy
import stl

from mlxtk.tools.gpop import transform_to_momentum_space


def plot_gpop(
    ax,
    time,
    grid,
    density,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    logz: bool = False,
):
    Y, X = numpy.meshgrid(grid, time)

    if logz:
        if zmin is not None:
            density[density < zmin] = zmin
        ret = ax.pcolormesh(
            X,
            Y,
            density,
            cmap="gnuplot",
            rasterized=True,
            norm=matplotlib.colors.LogNorm(zmin, zmax),
        )
    else:
        ret = ax.pcolormesh(X, Y, density, cmap="gnuplot", rasterized=True)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

    return ret


def plot_gpop_momentum(ax, time, grid, density):
    time, momentum, density = transform_to_momentum_space((time, grid, density))
    Y, X = numpy.meshgrid(momentum, time)
    ax.pcolormesh(X, Y, density, cmap="gnuplot", rasterized=True)
    ax.set_xlabel("$t$")
    ax.set_xlabel("$p$")


def create_gpop_model(time, grid, density):
    Y, X = numpy.meshgrid(grid / grid.max(), time / time.max())
    Z = density / density.max()

    fig, ax = matplotlib.pyplot.subplots(1, 1)
    ax.pcolormesh(X, Y, Z, cmap="gnuplot")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig("test.png", dpi=1200)
    matplotlib.pyplot.close()

    X = X.flatten()
    Y = Y.flatten()
    tri = matplotlib.tri.Triangulation(X, Y)
    Z = density.flatten()
    Z = Z / Z.max() * 0.1

    # , remove_empty_areas=False)
    mesh = stl.mesh.Mesh(numpy.zeros(len(tri.triangles), dtype=stl.mesh.Mesh.dtype))
    mesh.x[:] = X[tri.triangles]
    mesh.y[:] = Y[tri.triangles]
    mesh.z[:] = Z[tri.triangles]
    mesh.save("test.stl")
