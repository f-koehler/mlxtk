#!/usr/bin/env python
import mlxtk
from mlxtk import tasks
from mlxtk.systems.single_species.bose_hubbard import BoseHubbard

parameters = BoseHubbard.create_parameters()
parameters.N = 4
parameters.m = 4
parameters.sites = 4
parameters.pbc = True

if __name__ == "__main__":
    system = BoseHubbard(parameters)

    sim = mlxtk.Simulation("bose_hubbard")
    sim += tasks.CreateOperator(
        "fake_hamiltonian.opr",
        system.get_fake_initial_state_hamiltonian(),
    )
    sim += tasks.CreateMBOperator("hamiltonian.mb_opr", system.get_hamiltonian())
    sim += tasks.MCTDHBCreateWaveFunction(
        "initial.wfn",
        "fake_hamiltonian.opr",
        parameters.N,
        parameters.m,
        system.distribute_particles(),
    )
    sim += tasks.Relax(
        "gs_relax",
        "initial.wfn",
        "hamiltonian.mb_opr",
        tfinal=2000.0,
        dt=0.1,
        statsteps=50,
    )

    sim.main()
