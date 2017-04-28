from QDTK.Wavefunction import grab_lowest_eigenfct, Wavefunction
from QDTK.Tools.Mathematics import gramSchmidt

def init_single_bosonic_species(H, tape, num_spfs):
    evals, evecs = H.diag_1b_hamiltonian1d()
    spfs = grab_lowest_eigenfct(num_spfs, eves)
    gramSchmidt(spfs)
    wavefunction = Wavefunction(tape=tape)
    wavefunction.init_coef_sing_spc_B(
