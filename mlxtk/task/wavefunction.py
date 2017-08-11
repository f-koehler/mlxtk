from mlxtk import hashing
from mlxtk import log

import os
import sys
if sys.version_info <= (3, 0):
    from StringIO import StringIO
else:
    from io import StringIO


class WaveFunctionCreationTask:
    def __init__(self, project, name, func):
        self.project = project
        self.name = name
        self.func = func
        self.output_path = os.path.join(project.root_dir, name + ".wfn")
        self.wave_function_data = None
        self.logger = log.getLogger("operator")

    def is_up_to_date(self):
        # check for name conflict
        if self.name in self.project.wavefunctions:
            self.logger.critical("wave function \"%s\" already exists",
                                 self.name)
            raise RuntimeError(
                "name conflict: wave function \"{}\" already exists".format(
                    self.name))

        # calculate the operator and get the contents for the new operator file
        sio = StringIO()
        wavefunction = self.func()
        wavefunction.createWfnFile(sio)
        self.wave_function_data = sio.getvalue()

        # check if operator already up-to-date
        if os.path.exists(self.output_path):
            hash_current = hashing.hash_file(self.output_path)
            hash_new = hashing.hash_string(self.wave_function_data)
            if hash_current != hash_new:
                self.logger.info("hash mismatch")
                self.logger.debug("  %s != %s", hash_current, hash_new)
                return False
        else:
            self.logger.info("file does not exist")
            return False

        return True

    def execute(self):
        self.logger.info("task: create wave function \"%s\"", self.name)

        if self.is_up_to_date():
            self.project.wavefunctions[self.name] = {
                "updated": False,
                "path": self.output_path
            }
            self.logger.info("already up-to-date")
            self.logger.info("done")
            return

        # create the operator file
        self.logger.info("write operator file")
        with open(self.output_path, "w") as fh:
            fh.write(self.wave_function_data)

        # mark operator update
        self.project.wavefunctions[self.name] = {
            "updated": False,
            "path": self.output_path
        }

        self.logger.info("done")
