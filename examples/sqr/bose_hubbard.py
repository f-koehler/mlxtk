#!/usr/bin/env python
import mlxtk
from mlxtk import tasks
from mlxtk.systems.sqr.bose_hubbard import BoseHubbardSQR

parameters = BoseHubbardSQR.create_parameters()
parameters.N = 4
parameters.sites = 4
parameters.pbc = True
parameters.add_parameter("penalty", 100, "coefficient of the penalty term")

if __name__ == "__main__":
    system = BoseHubbardSQR(parameters)

    m = 5

    sim = mlxtk.Simulation("bose_hubbard")
    sim += tasks.CreateOperator("hamiltonian.opr", system.create_hamiltonian())
    sim += tasks.CreateSQRBosonicWaveFunction(
        "initial.wfn",
        [m, m + 2, m + 3, m + 4, m + 4],
        parameters.N,
        parameters.sites,
        [1, 1, 1, 1],
    )
    sim += tasks.Relax(
        "gs_relax",
        "initial.wfn",
        "hamiltonian.opr",
        tfinal=1200,
        dt=0.1,
        statsteps=50,
        exproj=False,
    )

    sim.main()
