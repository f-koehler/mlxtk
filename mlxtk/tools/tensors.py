from typing import Callable

import numpy

from mlxtk.dvr import DVRSpecification
from mlxtk.tools import ns_table


def get_delta_interaction_dvr(dvr: DVRSpecification, g: float = 1.0) -> numpy.ndarray:
    n = dvr.args[0]
    V = numpy.zeros((n, n, n, n), dtype=numpy.float64)
    weights = dvr.get_weights()

    for i in range(n):
        V[i, i, i, i] = g / weights[i]

    return V


def get_kinetic_spf(
    dvr: DVRSpecification, spfs: numpy.ndarray, prefactor: float = -0.5
) -> numpy.ndarray:
    m = spfs.shape[0]
    n = spfs.shape[1]
    spfs_c = numpy.conjugate(spfs)
    T_dvr = prefactor * dvr.get_d2()
    T = numpy.zeros((m, m), dtype=numpy.complex128)

    for a in range(m):
        for b in range(m):
            for i in range(n):
                for j in range(n):
                    T[a, b] += T_dvr[i, j] * spfs_c[a, i] * spfs[b, j]

    return T


def get_potential_spf(
    dvr: DVRSpecification,
    spfs: numpy.ndarray,
    potential: Callable[[numpy.ndarray], numpy.ndarray],
    prefactor: float = 1.0,
) -> numpy.ndarray:
    m = spfs.shape[0]
    spfs_c = numpy.conjugate(spfs)
    V_dvr = prefactor * potential(dvr.get_x())
    V = numpy.zeros((m, m), dtype=numpy.complex128)

    for a in range(m):
        for b in range(m):
            V[a, b] = numpy.sum(V_dvr * spfs_c[a] * spfs[b])

    return V


def get_delta_interaction_spf(
    dvr: DVRSpecification, spfs: numpy.ndarray, g: float = 1.0
) -> numpy.ndarray:
    m = spfs.shape[0]
    spfs_c = numpy.conjugate(spfs)
    V = numpy.zeros((m, m, m, m), dtype=numpy.complex128)
    w = dvr.get_weights()

    for a in range(m):
        for b in range(m):
            for c in range(m):
                for d in range(m):
                    V[a, b, c, d] = g * numpy.sum(
                        (1 / w) * spfs_c[a] * spfs_c[b] * spfs[c] * spfs[d]
                    )

    return V


def get_dmat_spf_naive(
    coefficients: numpy.ndarray, states: numpy.ndarray, normalize: bool = True
):
    m = len(states[0])
    N = numpy.sum(states[0])
    rho_1 = numpy.zeros((m, m), dtype=numpy.complex128)
    state_l = states[0].copy()
    state_r = states[0].copy()

    for i in range(m):
        for j in range(m):
            for state1, coefficient1 in zip(states, coefficients):
                for state2, coefficient2 in zip(states, coefficients):
                    state_l[:] = state1
                    state_r[:] = state2
                    state_l[i] -= 1
                    state_r[j] -= 1
                    if state_l[i] < 0:
                        continue
                    if state_r[j] < 0:
                        continue
                    if not numpy.array_equal(state_l, state_r):
                        continue

                    rho_1[i, j] += (
                        numpy.conjugate(coefficient1)
                        * coefficient2
                        * numpy.sqrt(state1[i] * state2[j])
                    )

    if normalize:
        return rho_1 / N

    return rho_1


def get_dmat_spf(
    coefficients: numpy.ndarray,
    states_Nm1: numpy.ndarray,
    states: numpy.ndarray,
    normalize: bool = True,
) -> numpy.ndarray:
    m = len(states_Nm1[0])
    N = numpy.sum(states[0])
    rho_1 = numpy.zeros((m, m), dtype=numpy.complex128)
    state_a = states_Nm1[0].copy()
    state_b = states_Nm1[0].copy()

    def get_index(ns: numpy.ndarray):
        for i, entry in enumerate(states):
            if numpy.array_equal(entry, ns):
                return i

        raise KeyError('ns "{}" not contained in number state table!'.format(ns))

    for i in range(m):
        for j in range(m):
            for state in states_Nm1:
                state_a[:] = state
                state_b[:] = state
                state_a[i] += 1
                state_b[j] += 1
                rho_1[i, j] += (
                    numpy.sqrt((state[i] + 1) * (state[j] + 1))
                    * numpy.conjugate(coefficients[get_index(state_a)])
                    * coefficients[get_index(state_b)]
                )

    if normalize:
        return rho_1 / N

    return rho_1


def get_dmat2_spf_naive(
    coefficients: numpy.ndarray, states: numpy.ndarray, normalize: bool = True
) -> numpy.ndarray:
    m = len(states[0])
    N = numpy.sum(states[0])
    rho_2 = numpy.zeros((m, m, m, m), dtype=numpy.complex128)
    state_l = states[0].copy()
    state_r = states[0].copy()

    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    for state1, coefficient1 in zip(states, coefficients):
                        for state2, coefficient2 in zip(states, coefficients):
                            state_l[:] = state1
                            state_r[:] = state2
                            state_l[i] -= 1
                            state_l[j] -= 1
                            state_r[k] -= 1
                            state_r[l] -= 1
                            if state_l[i] < 0:
                                continue
                            if state_l[j] < 0:
                                continue
                            if state_r[k] < 0:
                                continue
                            if state_r[l] < 0:
                                continue
                            if not numpy.array_equal(state_l, state_r):
                                continue

                            delta_l = 1 if (i == j) else 0
                            delta_r = 1 if (k == l) else 0

                            rho_2[i, j, k, l] += (
                                numpy.conjugate(coefficient1)
                                * coefficient2
                                * numpy.sqrt(
                                    (state1[i] - delta_l)
                                    * state1[j]
                                    * state2[k]
                                    * (state2[l] - delta_r)
                                )
                            )

    if normalize:
        return rho_2 / (N * (N - 1))

    return rho_2


def get_dmat2_spf(
    coefficients: numpy.ndarray,
    states_Nm2: numpy.ndarray,
    states: numpy.ndarray,
    normalize: bool = True,
) -> numpy.ndarray:
    m = len(states_Nm2[0])
    N = numpy.sum(states[0])
    rho_2 = numpy.zeros((m, m, m, m), dtype=numpy.complex128)
    state_a = states_Nm2[0].copy()
    state_b = states_Nm2[0].copy()

    def get_index(ns: numpy.ndarray):
        for i, entry in enumerate(states):
            if numpy.array_equal(entry, ns):
                return i

        raise KeyError('ns "{}" not contained in number state table!'.format(ns))

    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):
                    for state in states_Nm2:
                        state_a[:] = state
                        state_b[:] = state
                        delta_a = 1 if i == j else 0
                        delta_b = 1 if k == l else 0
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
                        rho_2[i, j, k, l] += (
                            numpy.sqrt(
                                (state[i] + 1) * factor_a * (state[j] + 1) * factor_b
                            )
                            * numpy.conjugate(coefficients[get_index(state_a)])
                            * coefficients[get_index(state_b)]
                        )

    if normalize:
        return rho_2 / (N * (N - 1))

    return rho_2
