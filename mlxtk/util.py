import shutil
import os

from .log import get_logger

LOGGER = get_logger(__name__)


def copy(src, dst, force=False):
    if (not force) and os.path.exists(dst):
        raise FileExistsError(dst)

    LOGGER.debug("copy: %s -> %s", src, dst)
    shutil.copy2(src, dst)


class TemporaryCopy(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.copy = False

    def __enter__(self):
        if not os.path.exists(self.dst):
            LOGGER.debug("create temporary copy: %s -> %s", self.src, self.dst)
            copy(self.src, self.dst)
            self.copy = True

    def __exit__(self, type, value, traceback):
        if self.copy:
            LOGGER.debug("remove temporary copy: %s", self.dst)
            os.remove(self.dst)
