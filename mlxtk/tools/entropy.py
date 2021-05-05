"""Compute entropy
"""
from typing import Union

import numpy

from mlxtk.log import get_logger


def compute_entropy(
    natpop: numpy.ndarray, normalize: bool = False
) -> Union[numpy.ndarray, numpy.float64]:
    """Compute the Boltzmann entropy from natural populations.

    The entropy is computed using the formula
    :math:`S_\\mathrm{B}=-\\sum\\limits_{i}\\lambda_i\\ln(\\lambda_i)`.

    Arguments:
        natpop (numpy.ndarray): one- or two-dimensional array containing
            natural populations

    Returns:
        Boltzmann entropy
    """
    if len(natpop.shape) == 1:
        result = 0.0
        for lam in natpop:
            if lam != 0.0:
                result -= lam * numpy.log(lam)

        if normalize:
            m = natpop.shape[0]
            if m == 1:
                raise ZeroDivisionError("cannot normalize entropy for m=1")
            else:
                S_max = numpy.log(m)
                result = result / S_max

        return result

    if len(natpop.shape) == 2:
        result = numpy.zeros(natpop.shape[0])
        for i in range(natpop.shape[0]):
            for lam in natpop[i]:
                if lam != 0.0:
                    result[i] -= lam * numpy.log(lam)

        if normalize:
            m = natpop.shape[1]
            if m == 1:
                raise ZeroDivisionError("cannot normalize entropy for m=1")
            else:
                S_max = numpy.log(m)
                result = result / S_max

        return result

    raise ValueError("natpop must be either 1- or 2-dimensional")
