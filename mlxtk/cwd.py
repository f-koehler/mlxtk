"""Change the working directory properly.
"""
import os
from pathlib import Path
from typing import Union

from . import log
from .util import make_path

LOGGER = log.get_logger(__name__)


class WorkingDir:
    """Context manager to change working directory without getting lost

    When entering this context, the working directory is changed to a
    requested location. On leaving the context, the original working directory
    is restored.

    Args:
        path (str): Path of the desired working directory

    Attributes:
        path (str): Path of the desired working directory
        initial_dir (str): Working directory before changing
    """

    def __init__(self, path: Union[str, Path]):
        self.initial_dir = Path.cwd().resolve()
        self.path = make_path(path).resolve()

    def __enter__(self):
        """Enter this context.
        """
        LOGGER.debug("set new cwd: %s", self.path)
        os.chdir(str(self.path))

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit this context.
        """
        del exc_type
        del exc_value
        del traceback
        LOGGER.debug("go back to old cwd: %s", self.initial_dir)
        os.chdir(str(self.initial_dir))
