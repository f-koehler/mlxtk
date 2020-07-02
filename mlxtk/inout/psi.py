import io
import re
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy

from mlxtk.tools.wave_function import get_spfs, load_wave_function

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


def read_psi_ascii(
    path: Union[str, Path]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    path = str(path)

    times: List[float] = []
    psis: List[List[complex]] = []
    tape: List[int] = []
    tape_finished = False
    with open(path) as fhandle:
        for line in fhandle:
            if line.startswith("$time"):
                tape_finished = True
                m = RE_TIME.match(fhandle.readline())
                if not m:
                    raise RuntimeError(
                        "Error extracting time point from label: {}".format(line)
                    )
                time = float(m.group(1))
                times.append(time)
            elif line.startswith("$psi"):
                tape_finished = True
                psis.append([])
                match = RE_ELEMENT.match(fhandle.readline())
                while match:
                    psis[-1].append(float(match.group(1)) + 1j * float(match.group(2)))
                    match = RE_ELEMENT.match(fhandle.readline())

            if not tape_finished:
                if line.startswith("$tape"):
                    continue

                if not line.strip():
                    continue

                tape.append(int(line))

    return (
        numpy.array(tape, dtype=numpy.int64),
        numpy.array(times),
        numpy.array(
            [numpy.array(psi).transpose() for psi in psis], dtype=numpy.complex128
        ),
    )


def read_psi_frame_ascii(
    path: Union[str, Path], index: int
) -> Tuple[numpy.ndarray, float, numpy.ndarray]:
    path = str(path)

    counter = -1
    times: List[float] = []
    psi: List[complex] = []
    tape: List[int] = []
    tape_finished = False
    with open(path) as fhandle:
        for line in fhandle:
            if line.startswith("$time"):
                tape_finished = True
                m = RE_TIME.match(fhandle.readline())
                if not m:
                    raise RuntimeError(
                        "Error extracting time point from label: {}".format(line)
                    )
                time = float(m.group(1))
                times.append(time)
            elif line.startswith("$psi"):
                counter += 1
                match = RE_ELEMENT.match(fhandle.readline())
                while match:
                    if counter == index:
                        psi.append(float(match.group(1)) + 1j * float(match.group(2)))
                    match = RE_ELEMENT.match(fhandle.readline())

            if not tape_finished:
                if line.startswith("$tape"):
                    continue

                if not line.strip():
                    continue

                tape.append(int(line))

    if not psi:
        raise KeyError("index {} is out of bounds".format(index))

    return (
        numpy.array(tape, dtype=numpy.int64),
        numpy.array(times[index]),
        numpy.array(psi, dtype=numpy.complex128),
    )


def read_psi_hdf5(path):
    with h5py.File(path, "r") as fptr:
        tape = fptr["tape"][:]
        time = fptr["time"][:]
        psis = fptr["psis"][:, :]
    return [tape, time, psis]


def write_psi_hdf5(path, data):
    tape, time, psis = data
    with h5py.File(path, "w") as fptr:
        dset = fptr.create_dataset(
            "tape", tape.shape, dtype=numpy.int64, compression="gzip"
        )
        dset[:] = tape

        dset = fptr.create_dataset(
            "time", time.shape, dtype=numpy.float64, compression="gzip"
        )
        dset[:] = time

        dset = fptr.create_dataset(
            "psis", psis.shape, dtype=numpy.complex128, compression="gzip"
        )
        dset[:, :] = psis[:, :]


def write_psi_ascii(path, data):
    tape, time, psis = data
    with open(path, "w") as fptr:
        fptr.write("$tape\n")
        fptr.writelines(("\t{}\n".format(entry) for entry in tape))
        for i, time in enumerate(time):
            fptr.write("\n$time\n\t{}  [au]\n$psi\n".format(time))
            fptr.writelines(
                (
                    " ({},{})\n".format(numpy.real(entry), numpy.imag(entry))
                    for entry in psis[i]
                )
            )
