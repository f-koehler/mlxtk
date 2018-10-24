import h5py
import numpy
import pandas


def read_expval_ascii(path):
    df = pandas.read_csv(path, delim_whitespace=True, names=["time", "real", "imag"])
    return (
        numpy.array(df["time"].values, dtype=numpy.float64),
        numpy.array(df["real"].values, dtype=numpy.complex128)
        + 1j * numpy.array(df["imag"].values, dtype=numpy.complex128),
    )


def write_expval_hdf5(path, data):
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "time", (len(data[0]),), dtype=numpy.float64, compression="gzip"
        )
        dset[:] = data[0]

        dset = fp.create_dataset(
            "values", data[1].shape, dtype=numpy.complex128, compression="gzip"
        )
        dset[:] = data[1]
