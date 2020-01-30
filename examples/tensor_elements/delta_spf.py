#!/usr/bin/env python
import numpy

import mlxtk
from mlxtk import dvr
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap
from mlxtk.tools.tensors import get_delta_interaction_spf

grid = dvr.add_harmdvr(201, 0.0, 0.75)
parameters = HarmonicTrap.create_parameters()
system = HarmonicTrap(parameters, grid)

_, spfs = system.get_hamiltonian_1b().diagonalize(12)
tensor = get_delta_interaction_spf(grid, spfs)
numpy.save("delta_spf.npy", tensor)
