import os

from . import log

logger = log.get_logger(__name__)
previous_directories = []


def change_dir(path: str):
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise RuntimeError("New working directory \"{}\" does not exist".format(path))
    if not os.path.isdir(path):
        raise RuntimeError(
            "New working directory \"{}\" is not a directory".format(path)
        )

    # store the old working directory
    old = os.path.abspath(os.getcwd())
    previous_directories.append(old)
    logger.debug("store old cwd: %s", old)

    logger.debug("set new cwd: %s", path)
    os.chdir(path)


def go_back():
    old = previous_directories.pop()
    logger.debug("go back to old cwd: %s", old)
    os.chdir(old)
