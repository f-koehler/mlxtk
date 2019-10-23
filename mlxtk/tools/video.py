import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Union

from ..cwd import WorkingDir
from ..util import make_path


def create_slideshow(files: List[Union[Path, str]],
                     output_file: Union[Path, str],
                     duration: float = 30.0):
    if not len(files):
        raise RuntimeError("No images specified!")

    files = [make_path(file).resolve() for file in files]
    output_file = make_path(output_file).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        with WorkingDir(tmp):
            max_number_len = len(str(len(files)))

            for i, file in enumerate(files):
                shutil.copy2(file,
                             ("{:0" + str(max_number_len) + "d}{}").format(
                                 i, file.suffix))

            cmd = [
                "ffmpeg", "-r",
                str(float(len(files)) / duration), "-i",
                "%0" + str(max_number_len) + "d" + files[0].suffix,
                "output.mp4"
            ]
            subprocess.run(cmd)

            shutil.copy2("output.mp4", output_file)
