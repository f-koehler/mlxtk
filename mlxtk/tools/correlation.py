import cmath
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy

from mlxtk.inout.dmat import read_dmat_gridrep_ascii, write_dmat_gridrep_hdf5
from mlxtk.inout.dmat2 import read_dmat2_gridrep_ascii
from mlxtk.log import get_logger
from mlxtk.util import make_path

LOGGER = get_logger(__name__)


def compute_g1(
    data_dmat: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    LOGGER.info("compute g1")

    time, x1, x2, dmat = data_dmat

    g1 = numpy.zeros_like(dmat)
    # TODO: improve this naive implementation (maybe numba can help)
    for i in range(len(time)):
        for j in range(len(x1)):
            for k in range(len(x2)):
                g1[i, j, k] = dmat[i, j, k] / cmath.sqrt(dmat[i, j, j] * dmat[i, k, k])
    LOGGER.info("finished computing g1")

    return time, x1, x2, g1


def compute_g1_diff(
    data_dmat: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    LOGGER.info("compute g1")

    time, x1, x2, dmat = data_dmat

    g1 = numpy.zeros_like(dmat)
    # TODO: improve this naive implementation (maybe numba can help)
    for i in range(len(time)):
        for j in range(len(x1)):
            for k in range(len(x2)):
                g1[i, j, k] = dmat[i, j, k] - cmath.sqrt(dmat[i, j, j]) * cmath.sqrt(
                    dmat[i, k, k]
                )
    LOGGER.info("finished computing g1")

    return time, x1, x2, g1


def compute_g2(
    data_dmat: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
    data_dmat2: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    LOGGER.info("compute g2")

    time, x1, x2, dmat = data_dmat
    time_, x1_, x2_, dmat2 = data_dmat2

    assert time.shape == time_.shape
    assert x1.shape == x1_.shape
    assert x2.shape == x2_.shape
    assert numpy.allclose(time, time_)
    assert numpy.allclose(x1, x1_)
    assert numpy.allclose(x2, x2_)

    g2 = numpy.zeros_like(dmat)
    # TODO: improve this naive implementation (maybe numba can help)
    for i in range(len(time)):
        for j in range(len(x1)):
            for k in range(len(x2)):
                g2[i, j, k] = dmat2[i, j, k] / (dmat[i, j, j] * dmat[i, k, k])
    LOGGER.info("finished computing g2")

    return time, x1, x2, g2


def compute_g2_diff(
    data_dmat: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
    data_dmat2: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    LOGGER.info("compute g2")

    time, x1, x2, dmat = data_dmat
    time_, x1_, x2_, dmat2 = data_dmat2

    assert time.shape == time_.shape
    assert x1.shape == x1_.shape
    assert x2.shape == x2_.shape
    assert numpy.allclose(time, time_)
    assert numpy.allclose(x1, x1_)
    assert numpy.allclose(x2, x2_)

    g2 = numpy.zeros_like(dmat)
    # TODO: improve this naive implementation (maybe numba can help)
    for i in range(len(time)):
        for j in range(len(x1)):
            for k in range(len(x2)):
                g2[i, j, k] = dmat2[i, j, k] - (dmat[i, j, j] * dmat[i, k, k])
    LOGGER.info("finished computing g2")

    return time, x1, x2, g2
