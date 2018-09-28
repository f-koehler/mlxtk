#!/usr/bin/env python
import mlxtk
from mlxtk.systems.single_species.harmonic_oscillator import HarmonicOscillator

if __name__ == "__main__":
    x = mlxtk.dvr.add_harmdvr(400, 1., 1.)

    system = HarmonicOscillator(g=0.1)

    sim = mlxtk.Simulation("harmonic_trap")
    sim += system.get_1b_hamiltonian("hamiltonian_1b")
    sim += system.get_mb_hamiltonian("hamiltonian")
    sim += system.get_mb_hamiltonian("hamiltonian_quenched", g=0.1, omega=0.9)
    sim += mlxtk.tasks.create_mctdhb_wave_function("initial", "hamiltonian_1b", 2, 3)
    sim += mlxtk.tasks.relax("gs_relax", "initial", "hamiltonian")
    sim += mlxtk.tasks.propagate(
        "propagate",
        "gs_relax/final",
        "hamiltonian_quenched",
        tfinal=10.,
        dt=0.1,
        psi=True,
        keep_psi=True,
    )
    sim += mlxtk.tasks.extract_psi("propagate/psi")

    sim.main()
