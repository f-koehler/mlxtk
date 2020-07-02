from typing import Dict, Tuple, Union

import numpy
from scipy import fftpack


def transform_to_momentum_space(
    data: Union[
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
        Tuple[numpy.ndarray, Dict[int, numpy.ndarray], Dict[int, numpy.ndarray]],
    ]
) -> Union[
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
    Tuple[numpy.ndarray, Dict[int, numpy.ndarray], Dict[int, numpy.ndarray]],
]:
    if isinstance(data[1], dict):
        new_grids = {}  # type: Dict[int, numpy.ndarray]
        new_densities = {}  # type: Dict[int, numpy.ndarray]
        for key in data[1]:
            dx = numpy.abs(data[1][key][1] - data[1][key][0])
            new_grids[key] = fftpack.fftshift(
                fftpack.fftfreq(len(data[1][key]), dx)
            ) / (2.0 * numpy.pi)
            new_densities[key] = numpy.abs(
                fftpack.fftshift(fftpack.fft(data[2][key], axis=1), axes=1)
            )
        return (data[0].copy(), new_grids, new_densities)
    elif isinstance(data[1], numpy.ndarray):
        dx = numpy.abs(data[1][1] - data[1][0])
        return (
            data[0].copy(),
            fftpack.fftshift(fftpack.fftfreq(len(data[1]), dx)) / (2.0 * numpy.pi),
            numpy.abs(fftpack.fftshift(fftpack.fft(data[2], axis=1), axes=1)),
        )
    else:
        raise ValueError("Bad data format")
