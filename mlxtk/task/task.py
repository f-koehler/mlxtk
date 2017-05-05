import mlxtk.hash
import mlxtk.log as log

import os.path
import shutil


class Task:

    def __init__(self, name):
        self.name = name

    def get_hash_path(self):
        return os.path.join("hashes", "task_" + self.name + ".hash")

    def read_hash_file(self):
        with open(self.get_hash_path()) as fh:
            return fh.read()

    def write_hash_file(self):
        with open(self.get_hash_path(), "w") as fh:
            fh.write(self.hash())

    def get_working_dir(self):
        return "task_" + self.name

    def create_working_dir(self):
        working_dir = self.get_working_dir()
        if os.path.exists(working_dir):
            log.info("Remove old temporary dir: %s", working_dir)
            shutil.rmtree(working_dir)
        log.info("Create temporary dir: %s", working_dir)
        os.mkdir(working_dir)

        return working_dir

    def clean(self):
        working_dir = self.get_working_dir()
        if os.path.exists(working_dir):
            log.info("Remove temporary dir: %s", working_dir)
            shutil.rmtree(working_dir)

    def show_header(self):
        print("")
        log.draw_box("TASK: {}".format(self.name))

    def __str__(self):
        return str(self.__dict__)

    def hash(self):
        return mlxtk.hash.hash_dict(self.__dict__)
