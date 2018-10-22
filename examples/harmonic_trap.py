#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(400, 1.0, 1.0)

    parameters = HarmonicTrap.create_parameters()
    parameters.g = 0.1
    parameters_quenched = HarmonicTrap.create_parameters()
    parameters_quenched.g = 0.2

    system = HarmonicTrap(parameters)
    system_quenched = HarmonicTrap(parameters_quenched)

    sim = mlxtk.Simulation("harmonic_trap")
    sim += mlxtk.tasks.create_operator("hamiltonian_1b", system.get_hamiltonian_1b())
    sim += system.get_mb_hamiltonian("hamiltonian")
    sim += system_quenched.get_mb_hamiltonian("hamiltonian_quenched")
    sim += mlxtk.tasks.create_mctdhb_wave_function("initial", "hamiltonian_1b", 2, 5)
    sim += mlxtk.tasks.improved_relax(
        "gs_relax", "initial", "hamiltonian_quenched", 1, tfinal=1000.0, dt=0.01
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
