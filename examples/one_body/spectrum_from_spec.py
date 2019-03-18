#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(512, 0.0, 1.)

    parameters = HarmonicTrap.create_parameters()
    parameters.m = 19

    parameters_quenched = parameters.copy()
    parameters_quenched.omega = 0.7

    system = HarmonicTrap(parameters, x)
    system_quenched = HarmonicTrap(parameters_quenched, x)

    sim = mlxtk.Simulation("spectrum_from_spec")

    sim += mlxtk.tasks.compute_spectrum(system.get_hamiltonian_1b(),
                                        parameters.m, "hamiltonian_1b")

    sim.main()
