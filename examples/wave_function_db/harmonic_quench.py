#!/usr/bin/env python
import mlxtk
import mlxtk.parameters
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(225, 0.0, 0.3)

    parameters = HarmonicTrap.create_parameters()
    parameters.m = 5
    parameters.N = 2

    def create_simulation(p):
        p_quenched = p.copy()
        p_quenched.g = 0.0

        system_quenched = HarmonicTrap(p_quenched, x)

        sim = mlxtk.Simulation("harmonic_trap")
        sim += mlxtk.tasks.CreateMBOperator(
            "hamiltonian_quenched",
            system_quenched.get_hamiltonian(),
        )
        sim += mlxtk.tasks.RequestWaveFunction("initial", p, "harmonic_gs.py")
        sim += mlxtk.tasks.Propagate(
            "propagate",
            "initial",
            "hamiltonian_quenched",
            tfinal=5.0,
            dt=0.05,
            psi=True,
        )

        return sim

    scan = mlxtk.ParameterScan(
        "harmonic_quench",
        create_simulation,
        mlxtk.parameters.generate_all(parameters, {"g": [0.01, 0.1, 1.0]}),
    )
    scan.main()
