import mlxtk.hash

import os.path
import sys

if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO

import QDTK.Tools.Mathematics
import QDTK.Wavefunction


def write_wavefunction(wavefunction, path):
    """Write a :class:`QDTK.Wavefunction.Wavefunction` to a file.

    The wave function is written to the specified path. If the file is already
    existent and its hash is compared to the given wave function. A write takes
    only place if the file contains another wave function.

    Args:
        wavefunction: wave function to write
        path: path that the wave function should be written to

    Returns:
        bool: ``True`` if the file was created/updated, ``False`` if the file was already up to date.
    """
    f = StringIO()
    wavefunction.createWfnFile(f)

    s = f.getvalue()

    if os.path.exists(path):
        h_new = mlxtk.hash.hash_string(s)
        h_old = mlxtk.hash.hash_file(path)

        if h_new == h_old:
            return False

    with open(path, "w") as fh:
        fh.write(s)

    return True


def init_single_bosonic_species(hamiltonian_1b, tape, num_spfs, number_state):
    evals, evecs = hamiltonian_1b.diag_1b_hamiltonian1d()
    spfs = QDTK.Wavefunction.grab_lowest_eigenfct(num_spfs, evecs)
    QDTK.Tools.Mathematics.gramSchmidt(spfs)
    wavefunction = QDTK.Wavefunction.Wavefunction(tape=tape)
    wavefunction.init_coef_sing_spec_B(number_state, spfs, full_spf=True)
    return wavefunction
