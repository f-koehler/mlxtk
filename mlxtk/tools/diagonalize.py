from typing import List, Tuple

import numpy
import scipy.linalg

import QDTK.Tools.Mathematics
import QDTK.Wavefunction

from .. import log

LOGGER = log.get_logger(__name__)


def diagonalize_1b_operator(matrix: numpy.ndarray, number_eigenfunctions: int
                            ) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
    """Diagonalize the supplied one-dimensional one-body hamiltonian

    Args:
        hamiltonian (QDTK.Operator): operator to diagonalize
        number_eigenfunctions (int): number of eigenvalues/eigenvectors to
            compute

    Returns:
        A tuple with the list of eigenvalues as the first element followed by
        the eigenvectors
    """
    eigenvalues, eigenvectors = scipy.linalg.eig(matrix)
    eigenvalues = numpy.real(eigenvalues)

    QDTK.Tools.Mathematics.sortEigValsVecs(eigenvalues, eigenvectors)

    eigenvectors = QDTK.Wavefunction.grab_lowest_eigenfct(
        number_eigenfunctions, eigenvectors)
    eigenvalues = eigenvalues[0:number_eigenfunctions]
    QDTK.Tools.Mathematics.gramSchmidt(eigenvectors)
    return (eigenvalues, eigenvectors)


def store_eigen_vectors(path: str, eigenvectors: List[numpy.ndarray]):
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
    converted = (numpy.floor(energies / tolerance).astype(numpy.int64) +
                 0.5) * tolerance
    unique = (
        numpy.unique(numpy.floor(energies / tolerance).astype(numpy.int64)) +
        0.5) * tolerance
    degeneracies = []
    for energy in unique:
        degeneracies.append(list(numpy.nonzero(converted == energy)[0]))
    return degeneracies


def split_bands(spfs, periodicity):
    num_bands = len(spfs) // periodicity
    if len(spfs) % periodicity != 0:
        LOGGER.warning("band %d seems to be incomplete", num_bands - 1)
        exit()

    for i in range(num_bands):
        yield spfs[i * periodicity:(i + 1) * periodicity]


def get_position_operator_in_spf_basis(spfs, x):
    operator = numpy.zeros((len(spfs), len(spfs)), dtype=numpy.complex128)
    for i, spf_i in enumerate(spfs):
        for j, spf_j in enumerate(spfs):
            operator[i, j] = numpy.dot(spf_i.conjugate(),
                                       numpy.multiply(x, spf_j))
    return operator


def get_translation_operator_in_spf_basis(spfs, periodicity, x):
    offset = numpy.abs(x - (x[0] + 1.0 / periodicity)).argmin()
    operator = numpy.zeros((len(spfs), len(spfs)), dtype=numpy.complex128)
    for i, spf_i in enumerate(spfs):
        for j, spf_j in enumerate(spfs):
            operator[i, j] = numpy.vdot(spf_i, numpy.roll(spf_j, offset))
    return operator


# def create_wannier_states(spfs, periodicity, x):
#     new_spfs = []
#     for band in split_bands(spfs, periodicity):
#         operator = get_position_operator_in_spf_basis(band, x)
#         evals, evecs = scipy.linalg.eigh(operator)
#         LOGGER.info("position operator eigenvalues: %s", str(evals))

#         for evec in evecs:
#             new_spfs.append(numpy.zeros_like(spfs[0]))
#             for val, spf in zip(evals, band):
#                 new_spfs[-1] += val * spf
#     return new_spfs

# def create_bloch_states(spfs, periodicity, x):
#     new_spfs = []
#     for band in split_bands(spfs, periodicity):
#         operator = get_translation_operator_in_spf_basis(band, periodicity, x)
#         evals, evecs = scipy.linalg.eigh(operator)
#         LOGGER.info("translation operator eigenvalues: %s", str(evals))

#         for evec in evecs:
#             new_spfs.append(numpy.zeros_like(spfs[0]))
#             for val, spf in zip(evals, band):
#                 new_spfs[-1] += val * spf
#     return new_spfs
