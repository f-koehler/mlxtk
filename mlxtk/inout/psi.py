import re
import numpy
import pandas

RE_TIME = re.compile(r"^\s+(.+)\s+\[au\]$")
RE_ELEMENT = re.compile(r"^\s*\((.+)\,(.+)\)$")


def read_psi_ascii(path):
    times = []
    psis = []
    with open(path) as fhandle:
        line = fhandle.readline()

        for line in fhandle:
            if line.startswith("$time"):
                time = float(RE_TIME.match(fhandle.readline()).group(1))
                times.append(time)
            elif line.startswith("$psi"):
                psis.append([])
                m = RE_ELEMENT.match(fhandle.readline())
                while m:
                    psis[-1].append([float(m.group(1)), float(m.group(2))])
                    m = RE_ELEMENT.match(fhandle.readline())

    psis = [numpy.array(psi).transpose() for psi in psis]
    converted = [
        pandas.DataFrame({
            "real": psi[0],
            "imaginary": psi[1]
        }) for psi in psis
    ]

    return numpy.array(times), converted
