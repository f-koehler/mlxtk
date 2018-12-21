import gzip
import io
import os

import numpy

from QDTK.Wavefunction import Wavefunction


def load_wave_function(path: str) -> Wavefunction:
    _, ext = os.path.splitext(path)
    if ext == ".gz":
        with gzip.open(path) as fp:
            with io.StringIO(fp.read().decode()) as sio:
                return Wavefunction(wfn_file=sio)

    return Wavefunction(wfn_file=path)


def save_wave_function(path: str, wfn: Wavefunction):
    if os.path.splitext(path)[1] == ".gz":
        with io.StringIO() as sio:
            wfn.createWfnFile(sio)
            with gzip.open(path, "wb") as fp:
                fp.write(sio.getvalue().encode())
                return

    wfn.createWfnFile(path)


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


def add_momentum_split(wfn: Wavefunction, momentum: float,
                       x0: float) -> Wavefunction:
    # pylint: disable=protected-access
    num_spfs = wfn.tree._subnodes[0]._dim  # type: int
    len_spfs = wfn.tree._subnodes[0]._phiLen  # type: int
    grid = wfn.tree._topNode._pgrid[0]

    phase = numpy.exp(
        1j * (2.0 * numpy.heaviside(grid - x0, 0.5) - 1.0) * momentum * grid)

    for i in range(0, num_spfs):
        start = wfn.tree._subnodes[0]._z0 + i * len_spfs
        stop = start + len_spfs
        wfn.PSI[start:stop] = phase * wfn.PSI[start:stop]

    return wfn
