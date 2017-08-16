import os
import shutil

from mlxtk import hashing


class CopyWavefunctionTask(object):
    def __init__(self, project, name, path):
        self.project = project
        self.name = name
        self.path = path
        self.logger = self.project.get_logger("copy_wavefunction")

    def execute(self):
        self.logger.info("task: create wave function \"%s\"", self.name)

        if self.is_up_to_date():
            self._update_project(False)
            self.logger.info("already up-to-date")
            self.logger.info("done")
            return

        if not self._is_original_present():
            self.logger.critical(
                "original wave function file does not exist, cannot copy")

        self.logger.info("copy wavefunction")
        shutil.copy2(self.path, self._get_output_path())

        # mark wave function update
        self._update_project(True)
        self.logger.info("done")

    def is_up_to_date(self):
        self._check_conflict()

        if not self._is_copy_present():
            return False

        if not self._is_original_present():
            return True

        orig_hash = hashing.hash_file(self.path)
        copy_hash = hashing.hash_file(self._get_output_path())
        if orig_hash != copy_hash:
            self.logger.info("hash mismatch")
            self.logger.debug("%s != %s", copy_hash, orig_hash)
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

    def _is_copy_present(self):
        ret = os.path.exists(self._get_output_path())
        if not ret:
            self.logger.info("copied wave function file does not exist")
        return ret

    def _is_original_present(self):
        ret = os.path.exists(self.path)
        if not ret:
            self.logger.warning("original wave function file does not exist")
        return ret

    def _update_project(self, updated):
        self.project.wavefunctions[self.name] = {
            "updated": updated,
            "path": self._get_output_path()
        }
