from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy
import pandas

from mlxtk.inout import tools
from mlxtk.util import make_path


def read_dmat_gridrep(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_dmat_gridrep_hdf5(path, interior_path)

    return read_dmat_gridrep_ascii(path)


def read_dmat_evals_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    path = make_path(path)
    with open(path, "r") as fp:
        m = len(fp.readline().strip().split()) - 1

    names = ["time"] + ["orbital_" + str(i) for i in range(m)]
    df = pandas.read_csv(path, delim_whitespace=True, header=None, names=names)
    return df["time"].values, df[names[1:]].values


def add_dmat_evals_to_hdf5(
    fptr: Union[h5py.File, h5py.Group], time: numpy.ndarray, eigenvalues: numpy.ndarray
):
    group = fptr.create_group("eigenvalues")
    group.create_dataset("time", time.shape, time.dtype)[:] = time
    group.create_dataset("eigenvalues", eigenvalues.shape, eigenvalues.dtype)[
        :, :
    ] = eigenvalues


def read_dmat_evecs_grid_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    path = make_path(path)
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


def add_dmat_evecs_grid_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    times: numpy.ndarray,
    grid: numpy.ndarray,
    evecs: numpy.ndarray,
):
    group = fptr.create_group("evecs_grid")
    group.create_dataset("time", times.shape, times.dtype)[:] = times
    group.create_dataset("grid", grid.shape, grid.dtype)[:] = grid
    group.create_dataset("real", evecs.shape, evecs.real.dtype)[:, :, :] = evecs.real
    group.create_dataset("imag", evecs.shape, evecs.imag.dtype)[:, :, :] = evecs.imag


def read_dmat_evecs_spf_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    path = make_path(path)
    with open(path, "r") as fp:
        m = (len(fp.readline().strip().split()) - 2) // 2

    names = ["time", "index"]
    names_real = []
    names_imag = []
    for i in range(m):
        names.append("real_" + str(i))
        names.append("imag_" + str(i))
        names_real.append("real_" + str(i))
        names_imag.append("imag_" + str(i))
    df = pandas.read_csv(path, delim_whitespace=True, header=None, names=names)
    evecs = df[names_real].values + 1j * df[names_imag].values

    num_indices = 0
    t_0 = df["time"][0]
    for t in df["time"]:
        if t_0 != t:
            break
        num_indices += 1

    times = df["time"].values[::num_indices]

    return times, evecs.T.reshape(m, len(times), num_indices)


def add_dmat_evecs_spf_to_hdf5(
    fptr: Union[h5py.File, h5py.Group], time: numpy.array, evecs: numpy.ndarray
):
    group = fptr.create_group("evecs_spf")
    group.create_dataset("time", time.shape, time.dtype)[:] = time
    group.create_dataset("real", evecs.shape, evecs.real.dtype)[:, :, :] = evecs.real
    group.create_dataset("imag", evecs.shape, evecs.imag.dtype)[:, :, :] = evecs.imag


def read_dmat_spfrep_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    path = make_path(path)
    df = pandas.read_csv(
        str(path),
        delim_whitespace=True,
        header=None,
        names=["time", "i", "j", "real", "imag"],
    )

    time = numpy.unique(df["time"].values)
    num_times = len(time)
    num_i = len(numpy.unique(df["i"].values))
    num_j = len(numpy.unique(df["j"].values))
    elements = df["real"].values + 1j * df["imag"].values

    return time, elements.reshape((num_times, num_i, num_j))


def add_dmat_spfrep_to_hdf5(
    fptr: Union[h5py.File, h5py.Group], time: numpy.ndarray, dmat: numpy.ndarray
):
    group = fptr.create_group("dmat_spfrep")
    group.create_dataset("time", time.shape, time.dtype)[:] = time
    group.create_dataset("real", dmat.shape, dmat.real.dtype)[:, :, :] = dmat.real
    group.create_dataset("imag", dmat.shape, dmat.imag.dtype)[:, :, :] = dmat.imag


def read_dmat_gridrep_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(
        path,
        header=None,
        names=["time", "x1", "x2", "real", "imag"],
        delim_whitespace=True,
    )

    time = numpy.unique(df["time"].values)
    x1 = numpy.unique(df["x1"].values)
    x2 = numpy.unique(df["x2"].values)
    dmat2 = numpy.reshape(
        df["real"].values + 1j * df["imag"].values, (len(time), len(x1), len(x2))
    )
    return time, x1, x2, dmat2


def read_dmat_gridrep_hdf5(
    path: Union[str, Path], interior_path: str = "/"
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (
            fptr[interior_path]["time"][:],
            fptr[interior_path]["x1"][:],
            fptr[interior_path]["x2"][:],
            fptr[interior_path]["real"][:, :, :]
            + 1j * fptr[interior_path]["imag"][:, :, :],
        )


def read_dmat_spfrep_hdf5(
    path: Union[str, Path], interior_path: str = "/"
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (
            fptr[interior_path]["time"][:],
            fptr[interior_path]["real"][:, :, :]
            + 1j * fptr[interior_path]["imag"][:, :, :],
        )


def write_dmat_gridrep_ascii(
    path: Union[str, Path],
    data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
):
    dmat = data[3].flatten()

    df = pandas.DataFrame(
        data=numpy.c_[
            numpy.repeat(data[0], len(data[1]) * len(data[2])),
            numpy.tile(numpy.repeat(data[1], len(data[2])), len(data[0])),
            numpy.tile(data[2], len(data[0]) * len(data[1])),
            dmat.real,
            dmat.imag,
        ],
        columns=["time", "x1", "x2", "real", "imag"],
    )
    df.to_csv(path, header=False, index=False, sep="\t")


def add_dmat_gridrep_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
):
    group = fptr.create_group("dmat_gridrep")
    group.create_dataset("time", data[0].shape, dtype=numpy.float64)[:] = data[0]
    group.create_dataset("x1", data[1].shape, dtype=numpy.float64)[:] = data[1]
    group.create_dataset("x2", data[2].shape, dtype=numpy.float64)[:] = data[2]
    group.create_dataset("real", data[3].shape, dtype=numpy.float64)[:, :, :] = data[
        3
    ].real
    group.create_dataset("imag", data[3].shape, dtype=numpy.float64)[:, :, :] = data[
        3
    ].imag


def write_dmat_gridrep_hdf5(
    path: Union[str, Path],
    data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
):
    with h5py.File(path, "w") as fptr:
        add_dmat_gridrep_to_hdf5(fptr, data)
