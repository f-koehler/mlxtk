import numpy

import QDTK.Tools.Mathematics
import QDTK.Wavefunction

from .. import log

LOGGER = log.get_logger(__name__)


def diagonalize_1b_hamiltonian(hamiltonian, number_eigenfunctions,
                               grid_points):
    """Diagonalize the supplied one-dimensional one-body hamiltonian

    Args:
        hamiltonian (QDTK.Operator): operator to diagonalize
        number_eigenfunctions (int): number of eigenvalues/eigenvectors to compute
        grid_points (int): the number of grid points

    Returns:
        A tuple with the list of eigenvalues as the first element followed by the eigenvectors
    """
    eigenvalues, eigenvectors = hamiltonian.diag_1b_hamiltonian1d(grid_points)
    LOGGER.info("%d loweste eigenvalues: %s", number_eigenfunctions,
                str(eigenvalues[:number_eigenfunctions]))

    eigenvectors = QDTK.Wavefunction.grab_lowest_eigenfct(
        number_eigenfunctions, eigenvectors)
    eigenvalues = eigenvalues[0:number_eigenfunctions]
    QDTK.Tools.Mathematics.gramSchmidt(eigenvectors)
    return eigenvalues, eigenvectors


def store_eigen_vectors(path, eigenvectors):
    """Write eigenvectors to a file

    Args:
        path (str): path of the file
        eigenvectors: the eigenvectors to save
    """
    numpy.savetxt(path, eigenvectors)


def find_degeneracies(energies, tolerance=1e-8):
    """Detect degeneracies in an eigenvalue spectrum

    Args:
        energies (numpy.ndarray): array containing the eigenvalues
        tolerance (float): tolerance when considering eigenvalues idenitcal

    Returns:
        list: A list of tuples containing the indices of equal eigenvalues.
    """
    converted = (numpy.floor(energies / tolerance).astype(numpy.int64) + 0.5
                 ) * tolerance
    unique = (
        numpy.unique(numpy.floor(energies / tolerance).astype(numpy.int64)) +
        0.5) * tolerance
    degeneracies = []
    for energy in unique:
        degeneracies.append(list(numpy.nonzero(converted == energy)[0]))
    return degeneracies
