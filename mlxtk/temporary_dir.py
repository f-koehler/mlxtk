import shutil
from pathlib import Path
from typing import Union

from mlxtk.log import get_logger
from mlxtk.util import make_path


class TemporaryDir:
    def __init__(self, path: Union[str, Path]):
        self.logger = get_logger(__name__ + ".TemporaryDir")
        self.complete = False
        self.path = make_path(path).resolve()
        self.already_present = self.path.exists()

    def __enter__(self):
        if not self.already_present:
            self.logger.info("create tempory dir: %s", str(self.path))
            self.path.mkdir()

        return self

    def __exit__(self, etype, evalue, etrace):
        del etype
        del evalue
        del etrace

        if self.complete:
            self.logger.info("remove temporary dir: %s", str(self.path))
            shutil.rmtree(self.path)
