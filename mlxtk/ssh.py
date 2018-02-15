import getpass
import os.path
import shutil
import sys

from . import log
from . import process

LOGGER = log.get_logger(__name__)


class SSHPassword(object):
    def __init__(self):
        self.password = getpass.getpass("Please enter your SSH password: ")

    def __enter__(self):
        return self.password

    def __exit__(self, type, value, traceback):
        del self.password
        self.password = ""
