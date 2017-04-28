parameters = {
    "grid_points": 256,
    "x_min": -1,
    "x_max": 1,
    "w0": 1.05,
    "w1": 1.,
    "g": 2.,
    "N": 2,
    "num_spfs": 5,
    "number_state": [5, 0, 0, 0, 0]
}

grids = {
}

H1 = get_hamiltonian1()
H2 = get_hamiltonian2()
x = get_operator_x()

psi = init_wavefunction()
psi_relaxed = relax_wavefunction(psi, H)
psi2 = propagate_wavefunction(psi_relaxed, H2)
mean_x = measure(psi2, x)
