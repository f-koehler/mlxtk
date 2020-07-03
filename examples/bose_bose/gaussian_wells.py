#!/usr/bin/env python
import mlxtk
from mlxtk import tasks
from mlxtk.systems.bose_bose.two_gaussian_traps import TwoGaussianTraps

grid = mlxtk.dvr.add_sinedvr(300, -10, 10)

parameters = TwoGaussianTraps.create_parameters()
parameters.N_A = 2
parameters.N_B = 2
parameters.M_A = 3
parameters.M_B = 3
parameters.m_A = 5
parameters.m_B = 5
parameters.V0L = 20.0
parameters.V0R = 20.0

if __name__ == "__main__":
    sim = mlxtk.Simulation("gaussian_wells")

    system = TwoGaussianTraps(parameters, grid)

    sim += tasks.CreateOperator(
        "hamiltonian_1b_A", system.get_hamiltonian_left_well_1b_A()
    )
    sim += tasks.CreateOperator(
        "hamiltonian_1b_B", system.get_hamiltonian_right_well_1b_B()
    )
    sim += tasks.CreateMBOperator(
        "hamiltonian_relax", system.get_hamiltonian_left_right()
    )
    sim += tasks.CreateMBOperator("hamiltonian", system.get_hamiltonian_colliding())
    sim += tasks.CreateBoseBoseWaveFunction(
        "initial",
        "hamiltonian_1b_A",
        "hamiltonian_1b_B",
        parameters.N_A,
        parameters.N_B,
        parameters.M_A,
        parameters.M_B,
        parameters.m_A,
        parameters.m_B,
    )
    sim += tasks.ImprovedRelax(
        "gs_relax", "initial", "hamiltonian_relax", 1, tfinal=10000.0, dt=0.01
    )
    sim += tasks.Propagate(
        "propagate", "gs_relax/final", "hamiltonian", tfinal=10.0, dt=0.05, psi=True
    )

    sim.main()
