import shutil
from pathlib import Path
from typing import Union

from .util import make_path


class TemporaryDir:
    def __init__(self, path):
        self.complete = False
        self.path = make_path(path).resolve()
        self.already_present = self.path.exists()

    def __enter__(self):
        if not self.already_present:
            self.path.mkdir()

        return self

    def __exit__(self, etype, evalue, etrace):
        del etype
        del evalue
        del etrace

        if self.complete:
            shutil.rmtree(self.path)
