import os.path

from ..log import get_logger
from .task import Task


class ProjectSPFBasis(Task):
    def __init__(self, basis: str, psi_or_restart: str):
        self.basis = basis
        self.psi_or_restart = psi_or_restart

        self.dirname = os.path.dirname(psi_or_restart)
        if not self.dirname:
            self.dirname = os.path.curdir

        self.logger = get_logger(__name__ + ".ProjectSPFBasis")
