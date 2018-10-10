#!/usr/bin/env python
import mlxtk
import mlxtk.parameters
from mlxtk.systems.single_species.harmonic_trap import HarmonicTrap


if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(400, 1., 1.)

    parameters = HarmonicTrap.create_parameters()
    parameters.g = 0.1
    parameters_quenched = HarmonicTrap.create_parameters()
    parameters_quenched.g = 0.2
    parameters.add_parameter("m", 1, "number of single particle functions")

    def create_simulation(p):
        system = HarmonicTrap(p)
        system_quenched = HarmonicTrap(parameters_quenched)

        sim = mlxtk.Simulation("")
        sim += system.get_1b_hamiltonian("hamiltonian_1b")
        sim += system.get_mb_hamiltonian("hamiltonian")
        sim += system_quenched.get_mb_hamiltonian("hamiltonian_quenched")
        sim += mlxtk.tasks.create_mctdhb_wave_function(
            "initial", "hamiltonian_1b", 2, 5
        )
        sim += mlxtk.tasks.improved_relax(
            "gs_relax", "initial", "hamiltonian", 1, tfinal=1000., dt=0.01
        )

        return sim

    scan = mlxtk.ParameterScan(
        "harmonic_trap_scan",
        create_simulation,
        mlxtk.parameters.generate_all(parameters, {"m": [1, 2, 3, 4, 5, 6]}),
    )
    scan.main()
