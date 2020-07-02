import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from mlxtk import inout
from mlxtk.cwd import WorkingDir
from mlxtk.inout.psi import read_first_frame
from mlxtk.log import get_logger
from mlxtk.tools.wave_function import load_wave_function

LOGGER = get_logger(__name__)


def main():
    # pylint: disable=protected-access

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basis",
        type=Path,
        help="wave function file containing the basis to project onto",
    )
    parser.add_argument(
        "psi",
        type=Path,
        help="wave function file containing the wave function to analyse",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="name of the output file (optional)"
    )
    args = parser.parse_args()

    output = args.output
    if not output:
        output = Path("{}_{}.fixed_ns.h5".format(args.psi.stem, args.basis.stem))

    output = output.resolve()

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        LOGGER.info("create a restart file from psi file")
        with open(Path(tmpdir) / "restart", "w") as fp:
            fp.write(read_first_frame(args.psi))

        LOGGER.info("copy psi file")
        shutil.copy2(args.psi, Path(tmpdir) / "psi")

        LOGGER.info("copy basis")
        shutil.copy2(args.basis, Path(tmpdir) / "basis")

        with WorkingDir(tmpdir):
            cmd = [
                "qdtk_analysis.x",
                "-fixed_ns",
                "-rst_bra",
                "basis",
                "-rst_ket",
                "restart",
                "-psi",
                "psi",
                "-save",
                "result",
            ]
            LOGGER.info("run qdtk_analysis.x: %s", " ".join(cmd))
            subprocess.check_output(cmd)

            times, real, imag = inout.read_fixed_ns_ascii("result")
            wfn = load_wave_function("basis")
            inout.write_fixed_ns_hdf5(
                "result.h5", times, real, imag, wfn._tape[1], wfn._tape[3]
            )
            LOGGER.info("copy result")
            shutil.copy2("result.h5", output)


if __name__ == "__main__":
    main()
