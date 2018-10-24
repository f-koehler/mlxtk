from typing import Tuple

import h5py
import numpy
import pandas

from . import tools


def read_expval(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_expval_hdf5(path, interior_path)

    return read_expval(path)


def read_expval_ascii(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    df = pandas.read_csv(path, delim_whitespace=True, names=["time", "real", "imag"])
    return (
        numpy.array(df["time"].values, dtype=numpy.float64),
        numpy.array(df["real"].values, dtype=numpy.complex128)
        + 1j * numpy.array(df["imag"].values, dtype=numpy.complex128),
    )


def read_expval_hdf5(
    path: str, interior_path: str
) -> Tuple[numpy.ndarray, numpy.ndarry]:
    with h5py.File(path, "r") as fp:
        return fp[interior_path]["time"][:], fp[interior_path]["values"][:]


def write_expval_hdf5(path: str, data: Tuple[numpy.ndarray, numpy.ndarray]):
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "time", (len(data[0]),), dtype=numpy.float64, compression="gzip"
        )
        dset[:] = data[0]

        dset = fp.create_dataset(
            "values", data[1].shape, dtype=numpy.complex128, compression="gzip"
        )
        dset[:] = data[1]
