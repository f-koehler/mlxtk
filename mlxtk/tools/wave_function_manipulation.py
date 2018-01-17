import numpy

from QDTK.Wavefunction import Wavefunction


def load_wave_function(path):
    wfn = Wavefunction(wfn_file=path)
    return wfn


def add_momentum(wfn, momentum):
    num_spfs = wfn.tree._subnodes[0]._dim
    len_spfs = wfn.tree._subnodes[0]._phiLen
    grid = wfn.tree._topNode._pgrid[0]

    phase = numpy.exp(-1j * momentum * grid)

    for i in range(0, num_spfs):
        start = wfn.tree._subnodes[0]._z0 + i * len_spfs
        stop = start + len_spfs
        wfn.PSI[start:stop] = phase * wfn.PSI[start:stop]

    return wfn
