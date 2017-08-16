import os

from mlxtk import hashing
from mlxtk.stringio import StringIO


class WaveFunctionCreationTask(object):
    def __init__(self, project, name, func):
        self.project = project
        self.name = name
        self.func = func
        self.wave_function_data = None
        self.logger = self.project.get_logger("wavefunction")

    def execute(self):
        self.logger.info("task: create wave function \"%s\"", self.name)

        if self.is_up_to_date():
            self._update_project(False)
            self.logger.info("already up-to-date")
            self.logger.info("done")
            return

        # create the wave function file
        self.logger.info("write wave function file")
        with open(self._get_output_path(), "w") as fh:
            fh.write(self.wave_function_data)

        # mark wave function update
        self._update_project(True)
        self.logger.info("done")

    def set_project_targets(self):
        self._check_conflict()
        self.project.wavefunctions[self.name] = {
            "path": self._get_output_path()
        }

    def is_up_to_date(self):
        self._check_conflict()
        self._get_wavefunction_data()

        if not self._is_file_present():
            return False

        # check if wave function already up-to-date
        hash_current = hashing.hash_file(self._get_output_path())
        hash_new = hashing.hash_string(self.wave_function_data)
        if hash_current != hash_new:
            self.logger.info("hash mismatch")
            self.logger.debug("%s != %s", hash_current, hash_new)
            return False

        return True

    def _check_conflict(self):
        if self.name in self.project.wavefunctions:
            self.logger.critical("wave function \"%s\" already exists",
                                 self.name)
            raise RuntimeError(
                "name conflict: wave function \"{}\" already exists".format(
                    self.name))

    def _get_output_path(self):
        return os.path.join(self.project.get_wave_function_dir(), self.name)

    def _get_wavefunction_data(self):
        sio = StringIO()
        wavefunction = self.func()
        wavefunction.createWfnFile(sio)
        self.wave_function_data = sio.getvalue()

    def _is_file_present(self):
        ret = os.path.exists(self._get_output_path())
        if not ret:
            self.logger.info("wave function file does not exist")
        return ret

    def _update_project(self, updated):
        self.project.wavefunctions[self.name] = {
            "updated": updated,
            "path": self._get_output_path()
        }
