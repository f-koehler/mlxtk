import re
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy

from mlxtk.inout.psi import read_psi_ascii
from mlxtk.log import get_logger
from mlxtk.util import make_path

LOGGER = get_logger(__name__)


def read_eigenbasis_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    path = make_path(path)
    path_energies = path / "eigenenergies"
    path_vectors = path / "eigenvectors"

    regex_energy = re.compile(
        r"^\s+\d+\s+\(([+-]?\d+\.\d+(?:[eE][+-]?\d+)?),([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\)$"
    )

    energies = []
    with open(path_energies) as fptr:
        for line in fptr:
            m = regex_energy.match(line)
            if not m:
                raise RuntimeError("Malformed eigenenergy line: " + line)
            energies.append(float(m.group(1)) + 1j * float(m.group(2)))
    energies = numpy.array(energies)

    tape, _, psi = read_psi_ascii(path_vectors)

    return energies, tape, psi


def add_eigenbasis_to_hdf5(
    fptr: Union[h5py.File, h5py.Group],
    energies: numpy.ndarray,
    tape: numpy.ndarray,
    psi: numpy.ndarray,
):
    grp = fptr.create_group("energies")
    grp.create_dataset("real", energies.shape, dtype=numpy.float64)[:] = energies.real
    grp.create_dataset("imag", energies.shape, dtype=numpy.float64)[:] = energies.imag
    fptr.create_dataset("tape", tape.shape, dtype=numpy.int64)[:] = tape
    grp = fptr.create_group("vectors")
    grp.create_dataset("real", psi.shape, dtype=numpy.float64)[:, :] = psi.real
    grp.create_dataset("imag", psi.shape, dtype=numpy.float64)[:, :] = psi.imag
