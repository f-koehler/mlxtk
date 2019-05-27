from typing import Tuple

import numpy
import pandas


def read_dmat_evals(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with open(path, "r") as fp:
        m = len(fp.readline().strip().split()) - 1

    names = ["time"] + ["orbital_" + str(i) for i in range(m)]
    df = pandas.read_csv(path, delim_whitespace=True, header=None, names=names)
    return df["time"].values, df[names[1:]].values


def read_dmat_evecs_grid(
        path: str) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with open(path, "r") as fp:
        m = (len(fp.readline().strip().split()) - 2) // 2

    names = ["time", "grid"]
    names_real = []
    names_imag = []
    for i in range(m):
        names.append("real_" + str(i))
        names.append("imag_" + str(i))
        names_real.append("real_" + str(i))
        names_imag.append("imag_" + str(i))
    df = pandas.read_csv(path, delim_whitespace=True, header=None, names=names)
    evecs = df[names_real].values + 1j * df[names_imag].values

    grid_size = 0
    t_0 = df["time"][0]
    for t in df["time"]:
        if t_0 != t:
            break
        grid_size += 1

    grid = df["grid"].values[:grid_size]
    times = df["time"].values[::grid_size]

    return times, grid, evecs.T.reshape(m, len(times), len(grid))
