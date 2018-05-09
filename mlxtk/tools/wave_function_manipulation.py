import numpy

from QDTK.Wavefunction import Wavefunction


def load_wave_function(path):
    """Load a wave function from a file

    Args:
        path (str): Path of the wave function file

    Returns:
        QDTK.Wavefunction: A wave function object created from the file
    """
    wfn = Wavefunction(wfn_file=path)
    return wfn


def add_momentum(wfn, momentum):
    """Add a momentum to a single-species wave function

    This is achieved by displacing all SPFs in momentum space.

    Args:
        wfn (QDTK.Wavefunction): Initial wave function
        momentum (float): Momentum to add to the SPFs

    Returns:
        QDTK.Wavefunction: Modified wave function
    """
    num_spfs = wfn.tree._subnodes[0]._dim
    len_spfs = wfn.tree._subnodes[0]._phiLen
    grid = wfn.tree._topNode._pgrid[0]

    phase = numpy.exp(-1j * momentum * grid)

    for i in range(0, num_spfs):
        start = wfn.tree._subnodes[0]._z0 + i * len_spfs
        stop = start + len_spfs
        wfn.PSI[start:stop] = phase * wfn.PSI[start:stop]

    return wfn
