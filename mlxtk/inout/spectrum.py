import h5py
import numpy


def read_spectrum(path: str):
    with h5py.File(path, "r") as fp:
        return fp["energies"][:], fp["spfs"][:, :]


def write_spectrum(path: str, energies: numpy.ndarray, spfs: numpy.ndarray):
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "energies",
            energies.shape,
            dtype=numpy.float64,
            compression="gzip")
        dset[:] = energies

        dset = fp.create_dataset(
            "spfs", spfs.shape, dtype=numpy.complex128, compression="gzip")
        dset[:, :] = spfs
