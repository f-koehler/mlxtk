#!/usr/bin/env python
from mlxtk.operator import *
from mlxtk.project import *
from mlxtk.primitive import *
from mlxtk.wavefunction import *
from mlxtk.task import *

grid_points = 32
x_min = -1
x_max = 1
w0 = 1.05
w1 = 1.
g = 2.
N = 2
num_spfs = 5
number_state = [N, 0, 0, 0, 0]

grid_x = Harmdvr(grid_points, x_min, x_max)

tape = (-10, N, 1, num_spfs, -1, 1, 1, 0, grid_points, -2)


def get_1b_hamiltonian():
    H = Operator()
    H.define_grids([grid_x])
    H.addLabel("dx^2", Term(grid_x.d2dvr))
    H.addLabel("x^2", Term(grid_x.x**2))
    H.addLabel("kin", Coef(-0.5))
    H.addLabel("pot", Coef(0.5 * (w0**2)))

    table = create_table("kin | 1 dx^2", "pot | 1 x^2")
    H.readTable(table)

    return H


def get_initial_hamiltonian():
    H = Operatorb()
    H.define_dofs_and_grids((1,), (grid_x,))
    H.addLabel("dx^2", Termb(grid_x.d2dvr))
    H.addLabel("x^2", Termb(grid_x.x**2))
    H.addLabel("delta", Termb(grid_x.delta_w()))
    H.addLabel("kin", Coefb(-0.5))
    H.addLabel("pot", Coefb(0.5 * (w0**2)))
    H.addLabel("int", Coefb(g))

    table = create_table("kin | 1 dx^2", "pot | 1 x^2", "int | {1:1} delta")
    H.readTableb(table)

    return H


def get_quenched_hamiltonian():
    H = Operatorb()
    H.define_dofs_and_grids((1,), (grid_x,))
    H.addLabel("dx^2", Termb(grid_x.d2dvr))
    H.addLabel("x^2", Termb(grid_x.x**2))
    H.addLabel("delta", Termb(grid_x.delta_w()))
    H.addLabel("kin", Coefb(-0.5))
    H.addLabel("pot", Coefb(0.5 * (w1**2)))
    H.addLabel("int", Coefb(g))

    table = create_table("kin | 1 dx^2", "pot | 1 x^2", "int | {1:1} delta")
    H.readTableb(table)

    return H


def get_x():
    x = Operatorb()
    x.define_dofs_and_grids((1,), (grid_x,))
    x.addLabel("x", Termb(grid_x.x))
    x.addLabel("prefactor", Coefb(1))
    x.readTableb("prefactor | 1 x")
    return x


def get_x2():
    x2 = Operatorb()
    x2.define_dofs_and_grids((1,), (grid_x,))
    x2.addLabel("x^2", Termb(grid_x.x**2))
    x2.addLabel("prefactor", Coefb(1))
    x2.readTableb("prefactor | 1 x^2")
    return x2


def get_initial_wavefunction():
    return init_single_bosonic_species(get_1b_hamiltonian(), tape, num_spfs,
                                       number_state)


relaxation = Relaxation(
    "initial", "relaxed", "H_initial", statsteps=100, tfinal=5, dt=0.01)
propagation = Propagation(
    "relaxed", "propagated", "H_quenched", tfinal=5, dt=0.01)
propagation.add_expectation_value("x")
propagation.add_expectation_value("x2")

project = Project("test_project")
project.add_operator("H_1b", get_1b_hamiltonian)
project.add_operator("H_initial", get_initial_hamiltonian)
project.add_operator("H_quenched", get_quenched_hamiltonian)
project.add_operator("x", get_x)
project.add_operator("x2", get_x2)
project.add_wavefunction("initial", get_initial_wavefunction)
project.add_task(relaxation)
project.add_task(propagation)
project.main()
