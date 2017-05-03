import mlxtk.hash

import logging
import os.path
import shutil


class Task:

    def __init__(self, name):
        self.name = name
        self.root_dir = None

    def get_hash_path(self):
        return os.path.join(self.root_dir, "task_hashes", self.name + ".hash")

    def read_hash_file(self):
        with open(self.get_hash_path()) as fh:
            return fh.read()

    def write_hash_file(self):
        with open(self.get_hash_path(), "w") as fh:
            fh.write(self.hash())

    def get_tmp_dir(self):
        return os.path.join(self.root_dir, "tmp_" + self.name)

    def create_tmp_dir(self):
        tmp_dir = self.get_tmp_dir()
        if os.path.exists(tmp_dir):
            logging.info("Remove old temporary dir: %s", tmp_dir)
            shutil.rmtree(tmp_dir)
        logging.info("Create temporary dir: %s", tmp_dir)
        os.mkdir(tmp_dir)

        return tmp_dir

    def clean(self):
        tmp_dir = self.get_tmp_dir()
        if os.path.exists(tmp_dir):
            logging.info("Remove temporary dir: %s", tmp_dir)
            shutil.rmtree(tmp_dir)

    def __str__(self):
        return str(self.__dict__)

    def hash(self):
        return mlxtk.hash.hash_dict(self.__dict__)
