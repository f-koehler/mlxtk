import numpy

from mlxtk.dvr import DVRSpecification


def get_delta_interaction_dvr(dvr: DVRSpecification,
                              g: float = 1.0) -> numpy.ndarray:
    n = dvr.args[0]
    V = numpy.zeros((n, n, n, n), dtype=numpy.float64)
    weights = dvr.get_weights()

    for i in range(n):
        V[i, i, i, i] = g / weights[i]

    return V


def get_delta_interaction_spf(dvr: DVRSpecification,
                              spfs: numpy.ndarray,
                              g: float = 1.0) -> numpy.ndarray:
    m = spfs.shape[0]
    spfs_c = numpy.conjugate(spfs)
    V = numpy.zeros((m, m), dtype=numpy.complex128)

    for a in range(m):
        for b in range(m):
            for c in range(m):
                for d in range(m):
                    V[a, b, c, d] = g * numpy.sum(
                        spfs_c[a] * spfs_c[b] * spfs[c] * spfs[d])

    return V
