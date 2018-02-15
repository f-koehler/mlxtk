import os

from . import log


class PIDFile(object):
    def __init__(self, path):
        self.logger = log.get_logger(__name__)
        self.path = path

    def acquire(self):
        if self.exists():
            raise RuntimeError(
                "PID file \"" + self.path + "\" does exist, cannot run")

        self.logger.debug("create PID file \"" + self.path + "\"")
        with open(self.path, "w"):
            pass

    def release(self):
        if self.exists():
            self.logger.debug("remove PID file \"" + self.path + "\"")
            os.remove(self.path)
        else:
            self.logger.warn("PID file \"" + self.path +
                             "\"does not exist, cannot remove it")

    def exists(self):
        return os.path.exists(self.path)
