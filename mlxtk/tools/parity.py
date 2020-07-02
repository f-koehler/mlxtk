from typing import Tuple

import numpy

from mlxtk.tools.diagonalize import diagonalize_1b_operator


def compute_state_parity(state: numpy.ndarray) -> numpy.complex128:
    return numpy.vdot(state, state[::-1])


def compute_parity_operator(
    state1: numpy.ndarray, state2: numpy.ndarray
) -> numpy.ndarray:
    opr = numpy.zeros((2, 2), dtype=numpy.complex128)
    opr[0, 0] = compute_state_parity(state1)
    opr[0, 1] = numpy.vdot(state1, state2[::-1])
    opr[1, 0] = numpy.vdot(state2, state1[::-1])
    opr[1, 1] = compute_state_parity(state2)
    return opr


def compute_parity_eigenstates(
    state1: numpy.ndarray, state2: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    opr = compute_parity_operator(state1, state2)
    evals, evecs = diagonalize_1b_operator(opr, 2)
    return evals, numpy.array(evecs)
