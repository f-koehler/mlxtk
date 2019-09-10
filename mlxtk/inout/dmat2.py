from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy
import pandas

from ..util import make_path
from . import tools


def read_dmat2_gridrep(
        path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_dmat2_gridrep_hdf5(path, interior_path)

    return read_dmat2_gridrep_ascii(path)


def read_dmat2_gridrep(
        path: str
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        assert interior_path
        raise NotImplementedError()

    return read_dmat2_gridrep_ascii(path)


def read_dmat2_gridrep_ascii(
        path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(path,
                         header=None,
                         names=["time", "x1", "x2", "dmat2"],
                         delim_whitespace=True)

    time = numpy.unique(df["time"].values)
    x1 = numpy.unique(df["x1"].values)
    x2 = numpy.unique(df["x2"].values)
    dmat2 = numpy.reshape(df["dmat2"].values, (len(time), len(x1), len(x2)))
    return time, x1, x2, dmat2


def read_dmat2_gridrep_hdf5(
        path: Union[str, Path], interior_path: str = "/"
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return (fptr[interior_path]["time"][:], fptr[interior_path]["x1"][:],
                fptr[interior_path]["x2"][:],
                fptr[interior_path]["dmat2"][:, :, :])


def add_dmat2_gridrep_to_hdf5(
        fptr: Union[h5py.File, h5py.Group],
        data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
):
    fptr.create_dataset("time", data[0].shape,
                        dtype=numpy.float64)[:] = data[0]
    fptr.create_dataset("x1", data[1].shape, dtype=numpy.float64)[:] = data[1]
    fptr.create_dataset("x2", data[2].shape, dtype=numpy.float64)[:] = data[2]
    fptr.create_dataset("dmat2", data[3].shape,
                        dtype=numpy.float64)[:, :, :] = data[3]
