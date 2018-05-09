import numpy

import mlxtk

PARAMETERS = mlxtk.parameters.Parameters({
    "N": 2,
    "displacement": 0.0,
    "g": 0.1,
    "m": 5,
    "mass": 1.,
    "n": 255,
    "omega": 1.,
    "particle_type": "boson",
})

GRID_NAME = "x"


def potential(x, p):
    return 0.5 * p.mass * (p.omega**2) * ((x - p.displacement)**2)


def get_1b_hamiltonian(p):
    op = mlxtk.Operator()
    op.define_grids([mlxtk.dvr.get(GRID_NAME)])

    op.addLabel("dx^2", mlxtk.OTerm(mlxtk.dvr.get_d2(GRID_NAME)))
    op.addLabel("V", mlxtk.OTerm(potential(mlxtk.dvr.get_x(GRID_NAME), p)))

    op.addLabel("kin", mlxtk.OCoef(-1. / (2 * p.mass)))
    op.addLabel("pot", mlxtk.OCoef(0.5 * p.mass * (p.omega * p.omega)))

    table = "\n".join(["kin | 1 dx^2", "pot | 1 V"])
    op.readTable(table)

    return op


def get_hamiltonian(p):
    op = mlxtk.OperatorB()
    op.define_dofs_and_grids((1, ), (mlxtk.dvr.get(GRID_NAME)))

    op.addLabel("dx^2", mlxtk.OTermB(mlxtk.dvr.get_d2(GRID_NAME)))
    op.addLabel("V", mlxtk.OTermB(potential(mlxtk.dvr.get_x(GRID_NAME), p)))

    op.addLabel("kin", mlxtk.OCoefB(-1. / (2 * p.mass)))
    op.addLabel("pot", mlxtk.OCoefB(0.5 * p.mass * (p.omega * p.omega)))

    table = ["kin | 1 dx^2", "pot | 1 V"]

    if p.g != 0.0:
        op.addLabel("delta", mlxtk.OTermB(mlxtk.dvr.get(GRID_NAME).delta_w()))
        op.addLabel("int", mlxtk.OCoef(p.g))
        table.append("int | {1:1} delta")

    table = "\n".join(table)
    op.readTable(table)

    return op


def get_initial_wave_function(p):
    if p.particle_type == "boson":
        sym = 1
    elif p.particle_type == "fermion":
        sym = -1
    else:
        raise RuntimeError(
            "Particle type \"{}\" not supported (only \"boson\" and \"fermion\""
        )

    _, spfs = mlxtk.tools.diagonalize_1b_hamiltonian(
        get_1b_hamiltonian(p), p.m, p.n)

    tape = (-10, p.N, sym, p.m, -1, 1, 1, 0, p.n)
    ns = numpy.zeros(p.m)
    ns[0] = p.N

    wave_function = mlxtk.Wavefunction(tape=tape)
    wave_function.init_coef_sing_spec(ns, spfs, 1e-14, 1e-14, full_spf=True)

    return wave_function
