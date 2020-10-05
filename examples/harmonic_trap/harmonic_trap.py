#!/usr/bin/env python
import mlxtk
from mlxtk import tasks
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

grid = mlxtk.dvr.add_fft(675, -20, 20)

parameters = HarmonicTrap.create_parameters()
parameters.N = 2
parameters.m = 3
parameters.g = 0.1
parameters.x0 = 10.0

if __name__ == "__main__":

    parameters_quenched = parameters.copy()
    parameters_quenched.x0 = 0.0

    system = HarmonicTrap(parameters, grid)
    system_quenched = HarmonicTrap(parameters_quenched, grid)

    sim = mlxtk.Simulation("harmonic_trap")

    sim += tasks.CreateOperator("hamiltonian_1b.opr", system.get_hamiltonian_1b())
    sim += tasks.CreateMBOperator("hamiltonian.mb_opr", system.get_hamiltonian())
    sim += tasks.CreateMBOperator(
        "hamiltonian_quenched.mb_opr", system_quenched.get_hamiltonian()
    )
    sim += tasks.CreateMBOperator("com,mb_opr", system_quenched.get_com_operator())
    sim += tasks.CreateMBOperator(
        "com_2.mb_opr", system_quenched.get_com_operator_squared()
    )

    sim += tasks.MCTDHBCreateWaveFunction(
        "initial.wfn", "hamiltonian_1b.opr", parameters.N, parameters.m
    )
    sim += tasks.ImprovedRelax(
        "gs_relax", "initial.wfn", "hamiltonian.mb_opr", 1, tfinal=1000.0, dt=0.01
    )
    sim += tasks.Propagate(
        "propagate",
        "gs_relax/final.wfn",
        "hamiltonian_quenched.mb_opr",
        tfinal=10.0,
        dt=0.05,
        psi=True,
    )

    sim += tasks.ComputeExpectationValue("propagate/psi", "com.mb_opr")
    sim += tasks.ComputeExpectationValue("propagate/psi", "com_2.mb_opr")
    sim += tasks.ComputeVariance("propagate/com", "propagate/com_2")

    sim += tasks.MCTDHBMomentumDistribution(
        "propagate/psi", "hamiltonian.mb_opr", "initial.wfn", grid
    )

    sim += tasks.ComputeExpectationValueStatic("propagate/final.wfn", "com.mb_opr")

    sim += tasks.NumberStateAnalysisStatic("propagate/final.wfn", "initial.wfn")

    sim += tasks.MCTDHBCreateWaveFunction(
        "basis_ED.wfn", "hamiltonian_1b.opr", parameters.N, 20
    )
    sim += tasks.Diagonalize("ED", "basis_ED.wfn", "hamiltonian.mb_opr", 5)

    sim.main()
