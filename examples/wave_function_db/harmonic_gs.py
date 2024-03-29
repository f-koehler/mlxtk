#!/usr/bin/env python
import mlxtk
import mlxtk.parameters
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

x = mlxtk.dvr.add_harmdvr(225, 0.0, 0.3)

parameters = HarmonicTrap.create_parameters()
parameters.g = 0.1
parameters.m = 5
parameters.N = 2


def create_simulation(p: mlxtk.parameters.Parameters):
    p_quenched = p.copy()
    p_quenched.g = 0.2

    system = HarmonicTrap(p, x)

    sim = mlxtk.Simulation("harmonic_trap")
    sim += mlxtk.tasks.CreateOperator("hamiltonian_1b", system.get_hamiltonian_1b())
    sim += mlxtk.tasks.CreateMBOperator("hamiltonian", system.get_hamiltonian())
    sim += mlxtk.tasks.MCTDHBCreateWaveFunction("initial", "hamiltonian_1b", p.N, p.m)
    sim += mlxtk.tasks.ImprovedRelax(
        "gs_relax",
        "initial",
        "hamiltonian",
        "1",
        tfinal=1000.0,
        dt=0.01,
    )

    return sim


db = mlxtk.WaveFunctionDB(
    "harmonic_gs",
    "gs_relax/final.wfn",
    parameters,
    create_simulation,
)

if __name__ == "__main__":
    db.main()
