from pathlib import Path
from typing import Any, Dict, List, Union

import numpy
import scipy.special

from QDTK.Wavefunction import Wavefunction


def load_wave_function(path: Union[str, Path]) -> Wavefunction:
    return Wavefunction(wfn_file=str(path))


def save_wave_function(path: Union[str, Path], wfn: Wavefunction):
    wfn.createWfnFile(str(path))


def add_momentum(wfn: Wavefunction, momentum: float) -> Wavefunction:
    # pylint: disable=protected-access
    num_spfs = wfn.tree._subnodes[0]._dim  # type: int
    len_spfs = wfn.tree._subnodes[0]._phiLen  # type: int
    grid = wfn.tree._topNode._pgrid[0]

    phase = numpy.exp(1j * momentum * grid)

    for i in range(0, num_spfs):
        start = wfn.tree._subnodes[0]._z0 + i * len_spfs
        stop = start + len_spfs
        wfn.PSI[start:stop] = phase * wfn.PSI[start:stop]

    return wfn


def add_momentum_split(
    wfn: Wavefunction, momentum: float, x_center: float
) -> Wavefunction:
    # pylint: disable=protected-access
    num_spfs = wfn.tree._subnodes[0]._dim  # type: int
    len_spfs = wfn.tree._subnodes[0]._phiLen  # type: int
    grid = wfn.tree._topNode._pgrid[0]

    phase = numpy.exp(
        1j * (2.0 * numpy.heaviside(grid - x_center, 0.5) - 1.0) * momentum * grid
    )

    for i in range(0, num_spfs):
        start = wfn.tree._subnodes[0]._z0 + i * len_spfs
        stop = start + len_spfs
        wfn.PSI[start:stop] = phase * wfn.PSI[start:stop]

    return wfn


def get_spfs(wfn: Wavefunction) -> List[numpy.ndarray]:
    # pylint: disable=protected-access
    num_spfs = wfn.tree._subnodes[0]._dim  # type: int
    len_spfs = wfn.tree._subnodes[0]._phiLen  # type: int
    spfs = []

    for i in range(num_spfs):
        start = wfn.tree._subnodes[0]._z0 + i * len_spfs
        stop = start + len_spfs
        spfs.append(wfn.PSI[start:stop])

    return spfs
