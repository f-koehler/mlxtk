from typing import Any, Dict

import numpy
import scipy.special

from mlxtk.util import memoize


@memoize
def binom(n: int, k: int) -> int:
    return int(scipy.special.binom(n, k))


# @jit
def build_number_state_table_bosonic(N: int, m: int) -> numpy.ndarray:
    number_of_states = binom(N + m - 1, m - 1)
    number_states = numpy.zeros((number_of_states, m), numpy.int64)

    number_states[0, 0] = N
    for i in range(number_of_states - 1):
        j = m - 2
        stop = False

        while j >= 0 and not stop:
            if number_states[i, j] > 0:
                summation = 0
                for k in range(j):
                    number_states[i + 1, k] = number_states[i, k]
                    summation += number_states[i + 1, k]
                number_states[i + 1, j] = number_states[i, j] - 1
                summation += number_states[i + 1, j]
                number_states[i + 1, j + 1] = N - summation
                stop = True
            j -= 1

    return number_states


class NumberStateLookupTableBosonic:
    def __init__(self, N: int, m: int):
        self.N = N
        self.m = m
        self.number_of_states = binom(N + m - 1, m - 1)
        self.table = {}  # type: Dict[int, Any]

        self._build()

    def _build(self):
        state_i = numpy.zeros(self.m, numpy.int64)
        state_ip1 = numpy.zeros(self.m, numpy.int64)

        state_i[0] = self.N
        self.insert_state(state_i, 0)

        for i in range(self.number_of_states - 1):
            j = self.m - 2
            stop = False

            while j >= 0 and not stop:
                if state_i[j] > 0:
                    summation = 0
                    for k in range(j):
                        state_ip1[k] = state_i[k]
                        summation += state_ip1[k]
                    state_ip1[j] = state_i[j] - 1
                    summation += state_ip1[j]
                    state_ip1[j + 1] = self.N - summation
                    stop = True
                j -= 1

            self.insert_state(state_ip1, i + 1)
            state_i[:] = state_ip1[:]
            state_ip1[:] = 0

    def insert_state(self, state: numpy.ndarray, index: int):
        self._insert_state_impl(state, index, 0, self.table)

    def _insert_state_impl(
        self, state: numpy.ndarray, index: int, position: int, current: Dict[int, Any]
    ):
        occupation = state[position]
        if position == state.shape[0] - 1:
            current[occupation] = index
            return

        if occupation not in current:
            current[occupation] = {}

        self._insert_state_impl(state, index, position + 1, current[occupation])

    def get_index(self, state: numpy.ndarray) -> int:
        return self._get_index_impl(state, 0, self.table)

    def _get_index_impl(
        self, state: numpy.ndarray, position: int, current: Dict[int, Any]
    ) -> int:
        occupation = state[position]
        if position == state.shape[0] - 1:
            return current[occupation]

        return self._get_index_impl(state, position + 1, current[occupation])


def get_number_state_index_bosonic(state: numpy.ndarray) -> int:
    index = 1
    m = state.shape[0]
    remaining = numpy.sum(state) - 1

    i = 0
    while i < m - 1:
        remaining -= state[i]
        i += 1

        if remaining > 0:
            j = 0
            while j <= remaining:
                index += binom(j + m - i - 1, m - i - 1)
                j += 1

    return index
