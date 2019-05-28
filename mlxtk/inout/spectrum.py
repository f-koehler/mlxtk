import h5py
import numpy


def read_spectrum(path: str):
    with h5py.File(path, "r") as fptr:
        return fptr["energies"][:], fptr["spfs"][:, :]


def write_spectrum(path: str, energies: numpy.ndarray, spfs: numpy.ndarray):
    with h5py.File(path, "w") as fptr:
        dset = fptr.create_dataset("energies",
                                   energies.shape,
                                   dtype=numpy.float64,
                                   compression="gzip")
        dset[:] = energies

        dset = fptr.create_dataset("spfs",
                                   spfs.shape,
                                   dtype=numpy.complex128,
                                   compression="gzip")
        dset[:, :] = spfs
