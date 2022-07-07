from __future__ import annotations

import multiprocessing
import os
import subprocess
from functools import reduce
from itertools import product
from operator import mul
from os import PathLike
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy
from numpy.typing import ArrayLike
from tqdm import tqdm

from mlxtk.cwd import WorkingDir
from mlxtk.dvr import DVRSpecification
from mlxtk.inout.expval import read_expval_ascii
from mlxtk.inout.psi import read_psi_frame_ascii, write_psi_ascii
from mlxtk.tasks.operator import OperatorSpecification
from mlxtk.temporary_dir import TemporaryDir
from mlxtk.util import copy_file


# function to compute one element of the reduced density matrix
def compute_reduced_density_matrix_element(
    dvrs: Mapping[int, DVRSpecification],
    dofs_A: Sequence[int],
    a: int,
    state_a: Sequence[ArrayLike],
    b: int,
    state_b: Sequence[ArrayLike],
) -> tuple[int, int, ArrayLike, ArrayLike]:
    # create projection operator |b><a| on the A system
    # unit operator acts on B for tracing it out
    operator_name = f"op_{a}_{b}.opr"
    operator = OperatorSpecification(
        dvrs,
        {"one": 1.0},
        {
            f"proj_{dof}": numpy.outer(
                numpy.conjugate(state_b[j]),
                state_a[j],
            )
            for j, dof in enumerate(dofs_A)
        },
        "one | " + " | ".join([f"{dof+1} proj_{dof}" for dof in dofs_A]),
    )
    operator.get_operator().createOperatorFile(operator_name)

    # compute expectation value
    output_name = f"ev_{a}_{b}.exp"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    subprocess.check_output(
        [
            "qdtk_expect.x",
            "-psi",
            "psi",
            "-rst",
            "restart",
            "-opr",
            operator_name,
            "-save",
            output_name,
        ],
        env=env,
    )

    # read and store result
    time, values = read_expval_ascii(output_name)

    Path(output_name).unlink()
    Path(operator_name).unlink()

    return a, b, time, values


def compute_reduced_density_matrix(
    wave_function: PathLike,
    output_file: PathLike,
    dvrs: Sequence[DVRSpecification],
    dofs_A: Sequence[int],
    basis_states: Mapping[int, Sequence[ArrayLike]] | None = None,
    progressbar: bool = False,
    threads: int = 1,
):
    # convert paths
    wave_function = Path(wave_function).resolve()
    output_file = Path(output_file).resolve()

    # function to generate basis states for DoF
    def generate_basis_states(dof: int) -> list[ArrayLike]:
        states: list[ArrayLike] = []
        dim = dvrs[dof].get().npoints
        for i in range(dim):
            states.append(numpy.zeros((dim,)))
            states[-1][i] = 1
        return states

    # check/prepare basis states for each DoF
    if basis_states is None:
        # generate basis_states from DVR
        basis_states: dict[int, list[ArrayLike]] = {}
        for dof in dofs_A:
            basis_states[dof] = generate_basis_states(dof)

    else:
        # check that basis_states are consitent with DVR
        for dof in dofs_A:
            # no basis states for DoF -> generate them
            if dof not in basis_states:
                basis_states[dof] = generate_basis_states(dof)

            dim = dvrs[dof].get().npoints

            if len(basis_states[dof]) != dim:
                raise ValueError(
                    f"basis for DoF {dof} is overcomplete or incomplete (got {len(basis_states[dof])} states, expected {dim})",
                )

            for i, state in enumerate(basis_states[dof]):
                if len(state) != dim:
                    raise ValueError(
                        f"basis state {i} of DoF {dof} has wrong size (got {len(state)}, expected {dim})",
                    )

    # compute dimension of reduced density operator
    dim_A = reduce(mul, (len(basis_states[dof]) for dof in basis_states))

    # create storage for results
    results: dict[tuple[int, int], complex] = {}

    with TemporaryDir(
        Path(output_file).parent / ("." + Path(output_file).name + ".tmp"),
    ) as tmpdir:
        with WorkingDir(tmpdir.path):
            # copy wave function file
            copy_file(Path(wave_function), Path.cwd() / "psi")

            # generate a restart file
            tape, times, psi = read_psi_frame_ascii("psi", 0)
            write_psi_ascii("restart", [tape, [times], psi[numpy.newaxis, :]])

            # generate the tasks for the multiprocessing pool
            tasks = (
                [dvrs, dofs_A, a, state_a, b, state_b]
                for a, state_a in enumerate(
                    product(*(basis_states[dof] for dof in dofs_A)),
                )
                for b, state_b in enumerate(
                    product(*(basis_states[dof] for dof in dofs_A)),
                )
                if b <= a
            )

            with multiprocessing.Pool(threads) as pool:
                for a, b, time, values in pool.starmap(
                    compute_reduced_density_matrix_element,
                    tasks,
                    1,
                ):
                    results[(a, b)] = values

        tmpdir.complete = True

    # convert results to an array
    steps = len(results[(0, 0)])
    results_arr = numpy.zeros((steps, dim_A, dim_A), dtype=numpy.complex128)
    for a in range(dim_A):
        for b in range(a + 1):
            results_arr[:, a, b] = results[(a, b)]

        for b in range(a + 1, dim_A):
            results_arr[:, a, b] = numpy.conjugate(results[(b, a)])

    # create parent directory of output file
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    # store results in a HDF5 file
    with h5py.File(output_file, "w") as fptr:
        fptr.create_dataset("time", data=time)
        fptr.create_dataset("rho_A", data=results_arr)
        fptr.create_dataset("trace", data=numpy.trace(results_arr, axis1=1, axis2=2))
        for dof in dofs_A:
            fptr.create_dataset(
                f"basis_states_{dof}",
                data=numpy.array(basis_states[dof]),
            )
