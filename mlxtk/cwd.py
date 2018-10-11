import os

from . import log

LOGGER = log.get_logger(__name__)


class WorkingDir:
    def __init__(self, path):
        self.initial_dir = os.path.abspath(os.getcwd())
        self.path = path

    def __enter__(self):
        LOGGER.debug("set new cwd: %s", self.path)
        os.chdir(self.path)

    def __exit__(self, type_, value, tb):
        LOGGER.debug("go back to old cwd: %s", self.initial_dir)
        os.chdir(self.initial_dir)
