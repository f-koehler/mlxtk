import numpy

from mlxtk.dvr import DVRSpecification
from mlxtk.tools import ns_table


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
    V = numpy.zeros((m, m, m, m), dtype=numpy.complex128)
    w = dvr.get_weights()

    for a in range(m):
        for b in range(m):
            for c in range(m):
                for d in range(m):
                    V[a, b, c, d] = g * numpy.sum(
                        (1 / w) * spfs_c[a] * spfs_c[b] * spfs[c] * spfs[d])

    return V


def get_dmat_spf(
        coefficients: numpy.ndarray, states_Nm1: numpy.ndarray,
        lookup: ns_table.NumberStateLookupTableBosonic) -> numpy.ndarray:
    m = len(states_Nm1[0])
    rho_1 = numpy.zeros((m, m), dtype=numpy.complex128)
    state_a = states_Nm1[0].copy()
    state_b = states_Nm1[0].copy()

    for i in range(m):
        for j in range(m):
            for state in states_Nm1:
                state_a = state.copy()
                state_b = state.copy()
                state_a[i] += 1
                state_b[j] += 1
                rho_1[i, j] = numpy.sqrt(
                    (state[i] + 1) * (state[j] + 1)) * numpy.conjugate(
                        coefficients[lookup.get_index(state_a)]
                    ) * coefficients[lookup.get_index(state_b)]

    return rho_1


def get_dmat2_spf(
        coefficients: numpy.ndarray, states_Nm2: numpy.ndarray,
        lookup: ns_table.NumberStateLookupTableBosonic) -> numpy.ndarray:
    m = len(states_Nm2[0])
    rho_2 = numpy.zeros((m, m, m, m), dtype=numpy.complex128)
    state_a = states_Nm2[0].copy()
    state_b = states_Nm2[0].copy()

    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    for state in states_Nm2:
                        state_a = state.copy()
                        state_b = state.copy()
                        delta_a = (1 if i == j else 0)
                        delta_b = (1 if k == l else 0)
                        factor_a = state[j] + 1 - delta_a
                        if factor_a <= 0:
                            continue
                        factor_b = state[l] + 1 - delta_b
                        if factor_b <= 0:
                            continue
                        state_a[i] += 1
                        state_a[j] += 1
                        state_b[k] += 1
                        state_b[l] += 1
                        rho_2[i, j, k, l] = numpy.sqrt(
                            (state[i] + 1) * factor_a *
                            (state[j] + 1) * factor_b) * numpy.conjugate(
                                coefficients[lookup.get_index(state_a)]
                            ) * coefficients[lookup.get_index(state_b)]

    return rho_2
