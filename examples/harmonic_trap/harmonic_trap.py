#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

grid = mlxtk.dvr.add_harmdvr(225, 0.0, 0.3)

parameters = HarmonicTrap.create_parameters()
parameters.N = 2
parameters.m = 3
parameters.g = 0.1

if __name__ == "__main__":

    parameters_quenched = parameters.copy()
    parameters_quenched.omega = 0.7

    system = HarmonicTrap(parameters, grid)
    system_quenched = HarmonicTrap(parameters_quenched, grid)

    sim = mlxtk.Simulation("harmonic_trap")

    sim += mlxtk.tasks.CreateOperator("hamiltonian_1b",
                                      system.get_hamiltonian_1b())
    sim += mlxtk.tasks.CreateMBOperator("hamiltonian",
                                        system.get_hamiltonian())
    sim += mlxtk.tasks.CreateMBOperator("hamiltonian_quenched",
                                        system_quenched.get_hamiltonian())
    sim += mlxtk.tasks.CreateMBOperator("com",
                                        system_quenched.get_com_operator())
    sim += mlxtk.tasks.CreateMBOperator(
        "com_2", system_quenched.get_com_operator_squared())

    sim += mlxtk.tasks.MCTDHBCreateWaveFunction("initial", "hamiltonian_1b",
                                                parameters.N, parameters.m)
    sim += mlxtk.tasks.ImprovedRelax("gs_relax",
                                     "initial",
                                     "hamiltonian",
                                     1,
                                     tfinal=1000.0,
                                     dt=0.01)
    sim += mlxtk.tasks.Propagate(
        "propagate",
        "gs_relax/final",
        "hamiltonian_quenched",
        tfinal=5.0,
        dt=0.05,
        psi=True,
    )

    sim += mlxtk.tasks.ComputeExpectationValue("propagate/psi", "com")
    sim += mlxtk.tasks.ComputeExpectationValue("propagate/psi", "com_2")
    sim += mlxtk.tasks.ComputeVariance("propagate/com", "propagate/com_2")

    sim += mlxtk.tasks.ComputeExpectationValueStatic("propagate/final", "com")

    sim += mlxtk.tasks.NumberStateAnalysisStatic("propagate/final", "initial")

    sim += mlxtk.tasks.MCTDHBCreateWaveFunction("basis_ED", "hamiltonian_1b",
                                                parameters.N, 20)
    sim += mlxtk.tasks.Diagonalize("ED", "basis_ED", "hamiltonian", 5)

    sim.main()
