from __future__ import annotations

import itertools
import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy

from mlxtk import cwd
from mlxtk.doit_compat import DoitAction
from mlxtk.hashing import inaccurate_hash
from mlxtk.inout.expval import read_expval_ascii
from numpy.typing import NDArray
from mlxtk.log import get_logger
from mlxtk.tasks.task import Task
from mlxtk.util import copy_file
from ..operator import OperatorSpecification


class ComputePointFunction(Task):
    def __init__(
        self,
        name: str,
        psi: os.PathLike,
        func: Callable[..., OperatorSpecification],
        indices: NDArray[numpy.int64],
        **kwargs,
    ):
        self.logger = get_logger(__name__ + ".ComputePointFunction")

        self.name = name
        self.psi = Path(psi)
        self.func = func
        self.indices = indices
        self.output = (self.psi.parent / name).with_suffix(".h5")
        self.pickle = (self.psi.parent / name).with_suffix(".pickle")

        self.wave_function = Path(
            kwargs.get("wave_function", self.psi.parent / "final"),
        ).with_suffix(".wfn")

    def task_write_parameters(self) -> dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: list[str]):
            del targets

            example = self.func(*self.indices[0])

            obj = [
                self.name,
                self.psi,
                self.indices,
                example.dofs,
                example.coefficients,
                {term: inaccurate_hash(example.terms[term]) for term in example.terms},
                example.table,
            ]

            with open(self.pickle, "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": f"point_function:{self.name}:write_parameters",
            "actions": [
                action_write_parameters,
            ],
            "targets": [
                str(self.pickle),
            ],
        }

    def task_compute(self) -> dict[str, Any]:
        @DoitAction
        def action_compute(targets: list[str]):
            del targets

            wave_function = self.wave_function.resolve()
            psi = self.psi.resolve()
            outpath = self.output.resolve()

            with tempfile.TemporaryDirectory() as tmpdir:
                with cwd.WorkingDir(tmpdir):
                    copy_file(psi, "psi")
                    copy_file(wave_function, "restart")

                    results: None | numpy.ndarray = None
                    cmd = [
                        "qdtk_expect.x",
                        "-opr",
                        "operator",
                        "-rst",
                        "restart",
                        "-psi",
                        "psi",
                        "-save",
                        "expval",
                    ]

                    for i, combination in enumerate(self.indices):
                        with open("operator", "w") as fptr:
                            self.func(*combination).get_operator().createOperatorFile(
                                fptr
                            )

                        self.logger.info(f'command: {" ".join(cmd)}')
                        env = os.environ.copy()
                        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                        subprocess.run(cmd, env=env)

                        time, values = read_expval_ascii("expval")
                        if results is None:
                            results = numpy.zeros(
                                [len(time), len(self.indices)], dtype=numpy.complex128
                            )
                        results[:, i] = values
                    with h5py.File(outpath, "w") as fptr:
                        fptr.create_dataset("time", data=time)
                        fptr.create_dataset("indices", data=self.indices)
                        fptr.create_dataset("values", data=results)

        return {
            "name": f"point_function:{self.name}:compute",
            "actions": [
                action_compute,
            ],
            "targets": [
                str(self.output),
            ],
            "file_dep": [str(self.pickle), str(self.psi)],
        }

    def get_tasks_run(self) -> list[Callable[[], dict[str, Any]]]:
        return [self.task_write_parameters, self.task_compute]
