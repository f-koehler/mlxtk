from pathlib import Path
from typing import Tuple, Union

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


def read_dmat_spfrep_ascii(path: Union[str, Path]):
    df = pandas.read_csv(str(path),
                         delim_whitespace=True,
                         header=None,
                         names=["time", "i", "j", "real", "imag"])

    time = numpy.unique(df["time"].values)
    num_times = len(time)
    num_i = len(numpy.unique(df["i"].values))
    num_j = len(numpy.unique(df["j"].values))
    elements = df["real"].values + 1j * df["imag"].values

    return time, elements.reshape((num_times, num_i, num_j))


def read_dmat_gridrep_ascii(
        path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(path,
                         header=None,
                         names=["time", "x1", "x2", "real", "imag"],
                         delim_whitespace=True)

    time = numpy.unique(df["time"].values)
    x1 = numpy.unique(df["x1"].values)
    x2 = numpy.unique(df["x2"].values)
    dmat2 = numpy.reshape(df["real"].values + 1j * df["imag"].values,
                          (len(time), len(x1), len(x2)))
    return time, x1, x2, dmat2


def write_dmat_gridrep_ascii(
        path: Union[str, Path],
        data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
):
    dmat = data[3].flatten()

    df = pandas.DataFrame(
        data=numpy.
        c_[numpy.repeat(data[0],
                        len(data[1]) * len(data[2])),
           numpy.tile(numpy.repeat(data[1], len(data[2])), len(data[0])),
           numpy.tile(data[2],
                      len(data[0]) * len(data[1])), dmat.real, dmat.imag],
        columns=["time", "x1", "x2", "real", "imag"])
    df.to_csv(path, header=False, index=False, sep="\t")
