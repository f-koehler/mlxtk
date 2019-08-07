from typing import Tuple

import h5py
import numpy
import pandas


def read_expval_ascii(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with open(path, "r") as fp:
        line = fp.readline().split()
        if len(line) == 2:
            return numpy.array([0.]), numpy.array(
                [float(line[0]) + 1j * float(line[1])])

    df = pandas.read_csv(path,
                         delim_whitespace=True,
                         names=["time", "real", "imag"])
    return (
        df["time"].values,
        df["real"].values + 1j * df["imag"].values,
    )


def read_expval_hdf5(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with h5py.File(path, "r") as fptr:
        return fptr["time"][:], fptr["real"][:] + 1j * fptr["imag"][:]


def write_expval_hdf5(path: str, time: numpy.ndarray, values: numpy.ndarray):
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset("time", time.shape, dtype=numpy.float64)
        dset[:] = time

        dset = fp.create_dataset("real", values.shape, dtype=numpy.float64)
        dset[:] = values.real

        dset = fp.create_dataset("imag", values.shape, dtype=numpy.float64)
        dset[:] = values.imag


def write_expval_ascii(path: str, data: Tuple[numpy.ndarray, numpy.ndarray]):
    pandas.DataFrame(
        numpy.column_stack((data[0], data[1].real, data[1].imag)),
        columns=["time", "real", "imag"],
    ).to_csv(path, sep="\t", index=False, header=False)
