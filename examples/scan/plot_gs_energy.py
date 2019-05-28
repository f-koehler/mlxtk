#!/usr/bin/env python
import os.path

import matplotlib.pyplot as plt
import numpy

import mlxtk
import mlxtk.inout.output

parameters = mlxtk.load_scan("harmonic_trap_scan")


def load_gs_energy(idx, path, parameter):
    data = mlxtk.inout.output.read_output(os.path.join(path,
                                                       "gs_relax/output"))
    return parameter.m, data[2][-1]


data = numpy.array(parameters.foreach(load_gs_energy))
plt.plot(data[:, 0], data[:, 1], marker=".")
plt.xlabel("$m$")
plt.ylabel("$E$")
plt.show()
