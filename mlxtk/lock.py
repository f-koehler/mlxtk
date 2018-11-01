import json
import os
import platform

from .log import get_logger

LOGGER = get_logger(__name__)


class LockFileExistsError(Exception):
    def __init__(self, path: str, host: str, pid: int):
        message = "Lock file {} exists (locked by PID {} on host {})".format(
            path, pid, host)
        super().__init__(message)


class LockFile:
    def __init__(self, path: str):
        self.path = os.path.realpath(os.path.abspath(path))

    def __enter__(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as fp:
                lock = json.load(fp)
                raise LockFileExistsError(self.path, lock["host"], lock["pid"])

        with open(self.path, "w") as fp:
            json.dump({"host": platform.node(), "pid": os.getpid()}, fp)

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type
        del exc_value
        del traceback

        os.remove(self.path)
