import io
import re
from typing import List, Tuple

import h5py
import numpy

from ..tools.wave_function import get_spfs, load_wave_function

RE_TIME = re.compile(r"^\s+(.+)\s+\[au\]$")
RE_ELEMENT = re.compile(r"^\s*\((.+)\,(.+)\)$")


def read_first_frame(path: str) -> str:
    frame = []  # type: List[str]
    encountered_time = False

    with open(path) as fhandle:
        for line in fhandle:
            if line.startswith("$time"):
                if encountered_time:
                    return "".join(frame)
                encountered_time = True
                frame.append(line)
                continue
            frame.append(line)

    return "".join(frame)


def read_spfs(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    with io.StringIO(read_first_frame(path)) as sio:
        wfn = load_wave_function(sio)

    _, times, psis = read_psi_ascii(path)

    spfs = []
    for psi in psis:
        wfn.PSI = psi
        spfs.append(numpy.array(get_spfs(wfn)))

    return times, numpy.moveaxis(numpy.array(spfs), 1, 0)


def read_psi_ascii(path: str
                   ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    times = []
    psis = []
    tape = []
    tape_finished = False
    with open(path) as fhandle:
        for line in fhandle:
            if line.startswith("$time"):
                tape_finished = True
                time = float(RE_TIME.match(fhandle.readline()).group(1))
                times.append(time)
            elif line.startswith("$psi"):
                psis.append([])
                match = RE_ELEMENT.match(fhandle.readline())
                while match:
                    psis[-1].append(
                        float(match.group(1)) + 1j * float(match.group(2)))
                    match = RE_ELEMENT.match(fhandle.readline())

            if not tape_finished:
                if line.startswith("$tape"):
                    continue

                if not line.strip():
                    continue

                tape.append(int(line))

    tape = numpy.array(tape, dtype=numpy.int64)
    times = numpy.array(times)
    psis = numpy.array([numpy.array(psi).transpose() for psi in psis],
                       dtype=numpy.complex128)

    return [tape, times, psis]


def read_psi_hdf5(path):
    with h5py.File(path, "r") as fptr:
        tape = fptr["tape"][:]
        time = fptr["time"][:]
        psis = fptr["psis"][:, :]
    return [tape, time, psis]


def write_psi_hdf5(path, data):
    tape, time, psis = data
    with h5py.File(path, "w") as fptr:
        dset = fptr.create_dataset("tape",
                                   tape.shape,
                                   dtype=numpy.int64,
                                   compression="gzip")
        dset[:] = tape

        dset = fptr.create_dataset("time",
                                   time.shape,
                                   dtype=numpy.float64,
                                   compression="gzip")
        dset[:] = time

        dset = fptr.create_dataset("psis",
                                   psis.shape,
                                   dtype=numpy.complex128,
                                   compression="gzip")
        dset[:, :] = psis[:, :]


def write_psi_ascii(path, data):
    tape, time, psis = data
    with open(path, "w") as fptr:
        fptr.write("$tape\n")
        fptr.writelines(("\t{}\n".format(entry) for entry in tape))
        for i, time in enumerate(time):
            fptr.write("\n$time\n\t{}  [au]\n$psi\n".format(time))
            fptr.writelines((" ({},{})\n".format(numpy.real(entry),
                                                 numpy.imag(entry))
                             for entry in psis[i]))
