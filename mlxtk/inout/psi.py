import re

import h5py
import numpy

RE_TIME = re.compile(r"^\s+(.+)\s+\[au\]$")
RE_ELEMENT = re.compile(r"^\s*\((.+)\,(.+)\)$")


def read_psi_ascii(path):
    times = []
    psis = []
    tape = []
    tape_finished = False
    with open(path) as fhandle:
        line = fhandle.readline()

        for line in fhandle:
            if line.startswith("$time"):
                tape_finished = True
                time = float(RE_TIME.match(fhandle.readline()).group(1))
                times.append(time)
            elif line.startswith("$psi"):
                psis.append([])
                m = RE_ELEMENT.match(fhandle.readline())
                while m:
                    psis[-1].append(float(m.group(1)) + 1j * float(m.group(2)))
                    m = RE_ELEMENT.match(fhandle.readline())

            if not tape_finished:
                if line.startswith("$tape"):
                    continue

                if not line.strip():
                    continue

                tape.append(int(line))

    tape = numpy.array(tape, dtype=numpy.int64)
    times = numpy.array(times)
    psis = numpy.array(
        [numpy.array(psi).transpose() for psi in psis], dtype=numpy.complex128)

    return [tape, times, psis]


def read_psi_hdf5(path):
    with h5py.File(path, "r") as fp:
        tape = fp["tape"][:]
        time = fp["time"][:]
        psis = fp["psis"][:, :]
    return [tape, time, psis]


def write_psi_hdf5(path, data):
    tape, time, psis = data
    with h5py.File(path, "w") as fp:
        dset = fp.create_dataset(
            "tape", tape.shape, dtype=numpy.int64, compression="gzip")
        dset[:] = tape

        dset = fp.create_dataset(
            "time", time.shape, dtype=numpy.float64, compression="gzip")
        dset[:] = time

        dset = fp.create_dataset(
            "psis", psis.shape, dtype=numpy.complex128, compression="gzip")
        dset[:, :] = psis[:, :]


def write_psi_ascii(path, data):
    tape, time, psis = data
    with open(path, "w") as fp:
        fp.write("$tape\n")
        fp.writelines(("\t{}\n".format(entry) for entry in tape))
        for i, time in enumerate(time):
            fp.write("\n$time\n\t{}  [au]\n$psi\n".format(time))
            fp.writelines((" ({},{})\n".format(
                numpy.real(entry), numpy.imag(entry)) for entry in psis[i]))
