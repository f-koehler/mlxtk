#!/usr/bin/env python
import collections
import os
import shutil
import urllib.request
from typing import Any, Dict, List

import jinja2
import numpy

import mlxtk

# if os.path.exists("report"):
#     shutil.rmtree("report")
# os.makedirs("report")


def merge(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(collections.ChainMap(*data))


def load_gs_energies(
    index: int,
    path: str,
    parameters: mlxtk.Parameters,
) -> Dict[str, float]:
    _, _, E, _ = mlxtk.inout.read_output(os.path.join(path, "rlx", "output"))
    return {parameters.gauge: E[-1]}


def load_gs_natpops(
    index: int,
    path: str,
    parameters: mlxtk.Parameters,
) -> Dict[str, List[float]]:
    _, natpops = mlxtk.inout.read_natpop(os.path.join(path, "rlx", "natpop"), 1, 1)
    return {parameters.gauge: natpops[-1].tolist()}


def load_gs_density(
    index: int,
    path: str,
    parameters: mlxtk.Parameters,
) -> Dict[str, numpy.ndarray]:
    _, _, gpop = mlxtk.inout.read_gpop(os.path.join(path, "rlx", "gpop"), 1)
    return {parameters.gauge: gpop[-1]}


def compute_density_error(density1: numpy.ndarray, density2: numpy.ndarray) -> float:
    return numpy.abs(density1 - density2).max()


selection = mlxtk.load_scan("ho_gauges")
data = {}
functions = {"compute_density_error": compute_density_error}

data["gs_energies"] = merge(selection.foreach(load_gs_energies))
data["gs_natpops"] = merge(selection.foreach(load_gs_natpops))
data["gs_density"] = merge(selection.foreach(load_gs_density))

with open(os.path.join("report", "index.html"), "w") as fp:
    fp.write(
        jinja2.Environment(loader=jinja2.FileSystemLoader("."))
        .get_template("report.html")
        .render(**data, **functions),
    )
