from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy
import pandas

from mlxtk.inout import tools
from mlxtk.util import make_path


def read_dmat2_gridrep(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_dmat2_gridrep_hdf5(path, interior_path)

    return read_dmat2_gridrep_ascii(path)


def read_dmat2_gridrep(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_dmat2_gridrep_hdf5(path, interior_path)

    return read_dmat2_gridrep_ascii(path)


def read_dmat2_gridrep_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(
        path, header=None, names=["time", "x1", "x2", "dmat"], delim_whitespace=True
    )

    time = numpy.unique(df["time"].values)
    x1 = numpy.unique(df["x1"].values)
    x2 = numpy.unique(df["x2"].values)
    dmat2 = numpy.reshape(df["dmat"].values, (len(time), len(x1), len(x2)))
    return time, x1, x2, dmat2


def read_dmat2_gridrep_hdf5(
    path: Union[str, Path], interior_path: str = "/"
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (
            fptr[interior_path]["time"][:],
            fptr[interior_path]["x1"][:],
            fptr[interior_path]["x2"][:],
            fptr[interior_path]["values"][:, :, :],
        )


def read_dmat2_spfrep_hdf5(
    path: Union[str, Path], interior_path: str = "/"
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (
            fptr[interior_path]["time"][:],
            fptr[interior_path]["real"][:, :, :]
            + 1j * fptr[interior_path]["imag"][:, :, :],
        )


def add_dmat2_gridrep_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
):
    group = fptr.create_group("dmat2_gridrep")
    group.create_dataset("time", data[0].shape, dtype=numpy.float64)[:] = data[0]
    group.create_dataset("x1", data[1].shape, dtype=numpy.float64)[:] = data[1]
    group.create_dataset("x2", data[2].shape, dtype=numpy.float64)[:] = data[2]
    group.create_dataset("values", data[3].shape, dtype=numpy.float64)[:, :, :] = data[
        3
    ]


def read_dmat2_spfrep_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(
        path,
        header=None,
        names=["time", "i", "j", "k", "l", "real", "imag"],
        delim_whitespace=True,
    )
    time = numpy.unique(df["time"].values)
    num_times = len(time)
    num_i = len(numpy.unique(df["i"].values))
    num_j = len(numpy.unique(df["j"].values))
    num_k = len(numpy.unique(df["k"].values))
    num_l = len(numpy.unique(df["l"].values))
    return (
        time,
        numpy.reshape(df["real"].values, (num_times, num_i, num_j, num_k, num_l))
        + 1j
        * numpy.reshape(df["imag"].values, (num_times, num_i, num_j, num_k, num_l)),
    )


def add_dmat2_spfrep_to_hdf5(
    fptr: [h5py.File, h5py.Group], time: numpy.ndarray, dmat2: numpy.ndarray
):
    group = fptr.create_group("dmat2_spfrep")
    group.create_dataset("time", shape=time.shape, dtype=time.dtype)[:] = time
    group.create_dataset("real", shape=dmat2.shape, dtype=dmat2.real.dtype)[
        :, :, :, :, :
    ] = dmat2.real
    group.create_dataset("imag", shape=dmat2.shape, dtype=dmat2.imag.dtype)[
        :, :, :, :, :
    ] = dmat2.imag
