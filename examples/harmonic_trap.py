#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(225, 0., 1.)
    # x = mlxtk.dvr.add_fft(225, -10., 10.)
    # x = mlxtk.dvr.add_expdvr(225, -10., 10.)

    parameters = HarmonicTrap.create_parameters()
    parameters.g = 0.0
    parameters_quenched = HarmonicTrap.create_parameters()
    parameters_quenched.g = 0.0

    system = HarmonicTrap(parameters, x)
    system_quenched = HarmonicTrap(parameters_quenched, x)

    sim = mlxtk.Simulation("harmonic_trap")
    sim += mlxtk.tasks.create_operator("hamiltonian_1b", system.get_hamiltonian_1b())
    sim += mlxtk.tasks.create_many_body_operator(
        "hamiltonian", system.get_hamiltonian()
    )
    sim += mlxtk.tasks.create_many_body_operator(
        "hamiltonian_quenched", system_quenched.get_hamiltonian()
    )
    sim += mlxtk.tasks.create_mctdhb_wave_function("initial", "hamiltonian_1b", 2, 5)
    sim += mlxtk.tasks.improved_relax(
        "gs_relax", "initial", "hamiltonian", 1, tfinal=1000.0, dt=0.01
    )
    sim += mlxtk.tasks.propagate(
        "propagate",
        "gs_relax/final",
        "hamiltonian_quenched",
        tfinal=10.0,
        dt=0.1,
        psi=True,
        keep_psi=True,
    )
    sim += mlxtk.tasks.extract_psi("propagate/psi")

    sim.main()
