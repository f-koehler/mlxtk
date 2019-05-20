#!/bin/bash
set -euf -o pipefail

cd "ho_gauges/by_param/gauge=standard/rlx/"
cp ../initial.wfn restart.ini
cp ../hamiltonian.mb_opr hamiltonian
qdtk_analysis.x -rst restart.ini -psi psi -opr hamiltonian -dmat -spfrep -diagonalize
# qdtk_analysis.x -rst restart.ini -psi psi -opr hamiltonian -dmat -dof 1 -gridrep -diagonalize
rm dmat_dof1_grid
