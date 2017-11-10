import numpy

import QDTK.Tools.Mathematics
import QDTK.Wavefunction


def diagonalize_1b_hamiltonian(hamiltonian, number_eigenfunctions,
                               grid_points):
    eigenvalues, eigenvectors = hamiltonian.diag_1b_hamiltonian1d(grid_points)
    eigenvectors = QDTK.Wavefunction.grab_lowest_eigenfct(
        number_eigenfunctions, eigenvectors)
    eigenvalues = eigenvalues[0:number_eigenfunctions]
    QDTK.Tools.Mathematics.gramSchmidt(eigenvectors)
    return eigenvalues, eigenvectors


def store_eigen_vectors(path, eigenvectors):
    numpy.savetxt(path, eigenvectors)
