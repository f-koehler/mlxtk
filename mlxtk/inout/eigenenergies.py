import h5py
import numpy

from . import tools


def read_eigenenergies(path: str):
    is_hdf5, path, interior_path = tools.is_hdf5_path(path)
    if is_hdf5:
        return read_eigenenergies_hdf5(path, interior_path)

    return read_eigenenergies_ascii(path)


def read_eigenenergies_ascii(path: str) -> numpy.ndarray:
    eigenenergies = []
    with open(path) as fp:
        for line in fp:
            values = line.split()[1].strip().lstrip("(").rstrip(")").split(",")
            eigenenergies.append(float(values[0]) + 1j * float(values[1]))
    return numpy.array(eigenenergies)


def read_eigenenergies_hdf5(path: str,
                            interior_path: str = "/") -> numpy.ndarray:
    with h5py.File(path, "r") as fp:
        return fp[interior_path]["eigenenergies"][:]


def write_eigenenergies_hdf5(path: str, data: numpy.ndarray):
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "eigenenergies",
            data.shape,
            dtype=numpy.complex128,
            compression="gzip")
        dset[:] = data
