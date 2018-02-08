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


def sync_to_node(path, node, password):
    if not shutil.which("rsync"):
        raise RuntimeError("Cannot find rsync executable")

    if not shutil.which("sshpass"):
        raise RuntimeError("Cannot find sshpass executable")

    if not os.path.isabs(path):
        raise RuntimeError("Path should be absolute")

    path = os.path.normpath(path) + os.path.sep

    LOGGER.info("rsync from node \"%s\": %s", node, path)

    cmd = ["sshpass", "-p", password]
    rsync_cmd = ["rsync", "--progress", "-avz", "-e", "ssh"]
    rsync_cmd.append(node + ":" + path)
    rsync_cmd.append(path)
    LOGGER.debug("command: %s", " ".join(rsync_cmd))

    process.watch_process(
        cmd + rsync_cmd,
        LOGGER.info,
        LOGGER.warn,
        stdout=sys.stdout,
        stderr=sys.stderr)

    LOGGER.info("complete")
