from __future__ import annotations
from mlxtk.tasks.task import Task
from mlxtk.log import get_logger
from mlxtk import cwd
from typing import Callable, Any
from pathlib import Path
import h5py
import os
from ..operator import OperatorSpecification
from mlxtk.doit_compat import DoitAction
from mlxtk.hashing import inaccurate_hash
import pickle
import numpy
import itertools
import tempfile
from mlxtk.util import copy_file
import subprocess
from mlxtk.inout.expval import read_expval_ascii


class ComputePointFunction(Task):
    def __init__(
        self,
        name: str,
        psi: os.PathLike,
        func: Callable[..., OperatorSpecification],
        start: int,
        stop: int,
        step: int = 1,
        num_points: int = 1,
        **kwargs,
    ):
        self.logger = get_logger(__name__ + ".ComputePointFunction")

        self.name = name
        self.psi = Path(psi)
        self.func = func
        self.start = start
        self.stop = stop
        self.step = step
        self.num_points = num_points
        self.output = (self.psi.parent / name).with_suffix(".h5")
        self.pickle = (self.psi.parent / name).with_suffix(".pickle")

        self.wave_function = Path(
            kwargs.get("wave_function", self.psi.parent / "final"),
        ).with_suffix(".wfn")

    def get_operator(self, indices: list[int]) -> OperatorSpecification:
        return self.func(*indices)

    def task_write_parameters(self) -> dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: list[str]):
            del targets

            example = self.get_operator(
                [self.start for index in range(self.num_points)]
            )

            obj = [
                self.name,
                self.psi,
                self.start,
                self.stop,
                self.step,
                self.num_points,
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
                    indices = numpy.arange(self.start, self.stop, self.step, dtype=int)
                    num_indices = len(indices)
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

                    for combination in itertools.product(
                        *([indices] * self.num_points)
                    ):
                        with open("operator", "w") as fptr:
                            self.get_operator(
                                combination
                            ).get_operator().createOperatorFile(fptr)

                        self.logger.info(f'command: {" ".join(cmd)}')
                        env = os.environ.copy()
                        env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                        subprocess.run(cmd, env=env)

                        time, values = read_expval_ascii("expval")
                        if results is None:
                            results = numpy.zeros(
                                [
                                    len(time),
                                ]
                                + [num_indices for i in range(self.num_points)],
                                dtype=numpy.complex128,
                            )

                        results[tuple([...] + list(combination))] = values[:]
                    with h5py.File(outpath, "w") as fptr:
                        fptr.create_dataset("time", data=time)
                        fptr.create_dataset("values", data=results)

        return {
            "name": f"point_function:{self.name}:compute",
            "actions": [
                action_compute,
            ],
            "targets": [
                str(self.output),
            ],
            "file_dep": [str(self.pickle)],
        }

    def get_tasks_run(self) -> list[Callable[[], dict[str, Any]]]:
        return [self.task_write_parameters, self.task_compute]
