import numpy
import pytest

from mlxtk import dvr
from mlxtk.tools import tensors


@pytest.mark.parametrize(
    "dvr",
    [
        dvr.add_harmdvr(201, 0.5, 1.1),
        dvr.add_expdvr(51, -10, 10),
        dvr.add_sinedvr(127, -3.0, 5.1),
        dvr.add_fft(201, -22.0, 17.0),
    ],
)
def test_delta_interaction_dvr(dvr: dvr.DVRSpecification):
    delta = tensors.get_delta_interaction_dvr(dvr)
    n = len(dvr.get_weights())

    diagonal = numpy.zeros(n, dtype=numpy.float64)
    for i in range(n):
        diagonal[i] = delta[i, i, i, i]

    assert numpy.allclose(diagonal, 1 / dvr.get_weights())
    assert numpy.allclose(diagonal, dvr.get_delta())
