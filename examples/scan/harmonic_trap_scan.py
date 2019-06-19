#!/usr/bin/env python
import mlxtk
import mlxtk.parameters
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(225, 0.0, 0.3)

    parameters = HarmonicTrap.create_parameters()
    parameters.g = 0.1
    parameters.N = 2
    parameters.m = 3

    def create_simulation(p):
        p_quenched = p.copy()
        p_quenched.g = 0.2

        system = HarmonicTrap(p, x)
        system_quenched = HarmonicTrap(p_quenched, x)

        sim = mlxtk.Simulation("harmonic_trap")
        sim += mlxtk.tasks.CreateOperator("hamiltonian_1b",
                                          system.get_hamiltonian_1b())
        sim += mlxtk.tasks.CreateMBOperator("hamiltonian",
                                            system.get_hamiltonian())
        sim += mlxtk.tasks.CreateMBOperator("hamiltonian_quenched",
                                            system_quenched.get_hamiltonian())
        sim += mlxtk.tasks.CreateMBOperator("com", system.get_com_operator())
        sim += mlxtk.tasks.CreateMBOperator("com_2",
                                            system.get_com_operator_squared())
        sim += mlxtk.tasks.MCTDHBCreateWaveFunction("initial",
                                                    "hamiltonian_1b", p.N, p.m)
        sim += mlxtk.tasks.ImprovedRelax("gs_relax",
                                         "initial",
                                         "hamiltonian",
                                         "1",
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

        return sim

    scan = mlxtk.ParameterScan(
        "harmonic_trap_scan",
        create_simulation,
        mlxtk.parameters.generate_all(
            parameters, {"g": [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}),
    )
    scan.main()
