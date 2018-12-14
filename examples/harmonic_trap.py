#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(225, 0.0, 0.3)

    parameters = HarmonicTrap.create_parameters()
    parameters.N = 10
    parameters.m = 3
    parameters.g = 0.5

    parameters_quenched = parameters.copy()
    parameters_quenched.omega = 0.7

    system = HarmonicTrap(parameters, x)
    system_quenched = HarmonicTrap(parameters_quenched, x)

    sim = mlxtk.Simulation("harmonic_trap")

    sim += mlxtk.tasks.create_operator("hamiltonian_1b",
                                       system.get_hamiltonian_1b())
    sim += mlxtk.tasks.create_mb_operator("hamiltonian",
                                          system.get_hamiltonian())
    sim += mlxtk.tasks.create_mb_operator("hamiltonian_quenched",
                                          system_quenched.get_hamiltonian())
    sim += mlxtk.tasks.create_mb_operator(
        "com", system_quenched.get_center_of_mass_operator())
    sim += mlxtk.tasks.create_mb_operator(
        "com_2", system_quenched.get_center_of_mass_operator_squared())

    sim += mlxtk.tasks.create_mctdhb_wave_function("initial", "hamiltonian_1b",
                                                   parameters.N, parameters.m)
    sim += mlxtk.tasks.improved_relax(
        "gs_relax", "initial", "hamiltonian", 1, tfinal=1000.0, dt=0.01)
    sim += mlxtk.tasks.propagate(
        "propagate",
        "gs_relax/final",
        "hamiltonian_quenched",
        tfinal=5.0,
        dt=0.05,
        psi=True,
        keep_psi=True, )

    sim += mlxtk.tasks.compute_expectation_value("propagate/psi", "com")
    sim += mlxtk.tasks.compute_expectation_value("propagate/psi", "com_2")
    sim += mlxtk.tasks.compute_variance("propagate/com", "propagate/com_2")

    sim.main()
