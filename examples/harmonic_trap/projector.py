#!/usr/bin/env python
import h5py
import numpy
from harmonic_trap import grid, parameters

from mlxtk.tasks.mb_operator import MBOperatorSpecification

if __name__ == "__main__":
    with h5py.File("gs_1b.h5", "r") as fptr:
        spf = fptr["wave_function"][:]

    projector = numpy.outer(spf, numpy.conjugate(spf))
    spec = MBOperatorSpecification(
        (1,),
        (grid,),
        {"coefficient": 1 / parameters.N},
        {"projector": projector},
        "coefficient | 1 projector",
    )

    with open("projector.mb_opr", "w") as fptr:
        spec.get_operator().createOperatorFileb(fptr)
