#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

x = mlxtk.dvr.add_harmdvr(225, 0.0, 0.3)

parameters = HarmonicTrap.create_parameters()
parameters.N = 3
parameters.m = 5
parameters.g = 1.5
parameters.add_parameter("gauge", "standard", "The gauge to use.")


def create_simulation(p):
    p_quenched = parameters.copy()
    p_quenched.omega = 0.1

    system = HarmonicTrap(p, x)
    system_quenched = HarmonicTrap(p_quenched, x)

    sim = mlxtk.Simulation("harmonic_trap")

    sim += mlxtk.tasks.CreateOperator("hamiltonian_1b",
                                      system.get_hamiltonian_1b())
    sim += mlxtk.tasks.CreateMBOperator("hamiltonian",
                                        system.get_hamiltonian())
    sim += mlxtk.tasks.CreateMBOperator("hamiltonian_quenched",
                                        system_quenched.get_hamiltonian())

    sim += mlxtk.tasks.MCTDHBCreateWaveFunction("initial", "hamiltonian_1b",
                                                p.N, p.m)
    sim += mlxtk.tasks.Propagate("propagate_0",
                                 "initial",
                                 "hamiltonian",
                                 tfinal=5.0,
                                 dt=0.05,
                                 psi=True,
                                 gauge=p.gauge)
    sim += mlxtk.tasks.Relax("rlx",
                             "initial",
                             "hamiltonian",
                             tfinal=1000.0,
                             dt=0.01,
                             psi=True,
                             gauge=p.gauge)
    sim += mlxtk.tasks.ImprovedRelax("imprlx",
                                     "initial",
                                     "hamiltonian",
                                     1,
                                     tfinal=1000.0,
                                     dt=0.01,
                                     psi=True,
                                     gauge=p.gauge)
    # sim += mlxtk.tasks.Propagate("propagate_rlx",
    #                              "rlx/final",
    #                              "hamiltonian_quenched",
    #                              tfinal=5.0,
    #                              dt=0.05,
    #                              psi=True,
    #                              gauge=p.gauge)
    # sim += mlxtk.tasks.Propagate(
    #     "propagate_imprlx",
    #     "imprlx/final",
    #     "hamiltonian_quenched",
    #     tfinal=5.0,
    #     dt=0.05,
    #     psi=True,
    #     gauge=p.gauge)

    return sim


scan = mlxtk.ParameterScan(
    "ho", create_simulation,
    mlxtk.parameters.generate_all(parameters, {"gauge": ["standard", "norb"]}))

if __name__ == "__main__":
    scan.main()
