"""Compatibility layer for the DoIt library.

This module provides various helper functions and classes to use the
`DoIt <http://pydoit.org/>`_ library in a way that is adequate for mlxtk.
"""
import json
import sqlite3
from typing import Any, Dict, List, Tuple

import doit
from doit.cmd_base import TaskLoader
from doit.doit_cmd import DoitMain
from doit.task import dict_to_task
from tabulate import tabulate

from .log import get_logger
from .timing import Timer

LOGGER = get_logger(__name__)


class CustomTaskLoader(TaskLoader):
    def __init__(self, task_generators):
        self.task_generators = task_generators

        super().__init__()

    def load_tasks(self, cmd, opt_values, pos_args):
        del cmd
        del opt_values
        del pos_args
        tasks = []
        for task_generator in self.task_generators:
            tasks.append(dict_to_task(task_generator()))
        return tasks, {}


def run_doit(task_generators, arguments=None) -> int:
    if arguments is None:
        raise ValueError("No arguments passed to DoitMain")
    LOGGER.debug("arguments for DoitMain: %s", str(arguments))
    LOGGER.debug("doit module: %s", doit.__file__)
    return DoitMain(CustomTaskLoader(task_generators)).run(arguments)


class DoitAction:
    def __init__(self, func):
        self.func = func

    def __call__(self, targets, *args, **kwargs):
        timer = Timer()
        ret = self.func(targets, *args, **kwargs)
        timer.stop()

        if ret is None:
            return {
                self.func.__name__: {
                    "monotonic_time": timer.get_monotonic_time(),
                    "perf_time": timer.get_perf_time(),
                    "process_time": timer.get_process_time(),
                }
            }

        if isinstance(ret, dict):
            ret[self.func.__name__] = ret.get(self.func.__name__, {})
            ret[self.func.
                __name__]["monotonic_time"] = timer.get_monotonic_time()
            ret[self.func.__name__]["perf_time"] = timer.get_perf_time()
            ret[self.func.__name__]["process_time"] = timer.get_process_time()
            return ret

        if isinstance(ret, bool):
            if not ret:
                return ret
            return {
                self.func.__name__: {
                    "monotonic_time": timer.get_monotonic_time(),
                    "perf_time": timer.get_perf_time(),
                    "process_time": timer.get_process_time(),
                }
            }

        raise NotImplementedError(
            "The return type {} is not supported for Doit actions")


def load_doit_db(path: str) -> Dict[str, Dict[str, Any]]:
    db = sqlite3.connect(path)
    cursor = db.cursor()
    cursor.execute("SELECT task_id, task_data FROM doit")
    result = cursor.fetchall()
    db.close()
    return {name: json.loads(data) for name, data in result}


def load_doit_timings(path: str) -> Dict[str, Dict[str, float]]:
    timings = {}  # type: Dict[str, Dict[str, float]]
    data = load_doit_db(path)
    for task in data:
        timings[task] = timings.get(task, {})
        for action in data[task]["_values_:"]:
            timings[task][action] = data[task]["_values_:"][action][
                "monotonic_time"]
    return timings


def load_doit_timing(path: str, task: str, action: str) -> float:
    return load_doit_timings(path)[task][action]


def format_doit_profile(timings: Dict[str, Dict[str, float]],
                        tablefmt: str = "fancy_grid") -> str:
    profile = []  # type: List[Tuple[str, str, float]]
    for task in timings:
        for action in timings[task]:
            profile.append((task, action, timings[task][action]))
    profile.sort(key=lambda x: x[2], reverse=True)
    return tabulate(profile,
                    headers=["Task", "Action", "Time/s"],
                    tablefmt=tablefmt)
