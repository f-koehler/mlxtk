import mlxtk.hash
import mlxtk.log as log
from mlxtk.filesystem import relative_symlink, unlink_if_present

import os.path
import shutil


class Task:
    def hash(self):
        """Compute the hash of the task

        Returns:
            str: hash of this instance which is computed by calling
                :meth:`mlxtk.hash.hash_dict` on ``__dict__``.
        """
        return mlxtk.hash.hash_dict(self.__dict__)

    def is_up_to_date(self):
        """ Check if the task is up-to-date

        First the hash value of the hash is checked. Afterwards the
        modification times of all the input files are checked.

        Raises:
            RuntimeError: If at least one of the input files does not exist
        """

        # check if all input files exist
        for input_file in self.input_files:
            if not os.path.exists(input_file):
                raise RuntimeError(
                    "Input file \"{}\" does not exist".format(input_file))

        # check if hash file exists
        if not os.path.exists(self.get_hash_path()):
            log.debug("Task %s is not up-to-date, hash file does not exist",
                      self.name)
            return False

        # check if hash has changed
        if self.hash() != self.read_hash_file():
            log.debug("Task %s is not up-to-date, hash differs from hash file",
                      self.name)
            return False

        # check if all output files exist
        for output_file in self.output_files:
            if not os.path.exists(
                    os.path.join(self.get_working_dir(), output_file)):
                log.debug(
                    "Task %s is not up-to-date, output file \"%s\" does not exist",
                    self.name, output_file)
                return False

        # check if any input file is newer than any output file
        newest_input = max(self.input_files,
                           key=lambda input_file: os.path.getmtime(input_file))
        oldest_output = min(self.output_files,
                            key=lambda output_file: os.path.getmtime(os.path.join(self.get_working_dir(), output_file)))
        if os.path.getmtime(newest_input) > os.path.getmtime(
                os.path.join(self.get_working_dir(), oldest_output)):
            log.debug(
                "Task %s is not up-to-date, input file \"%s\" is newer than output file \"%s\"",
                self.name, newest_input, oldest_output)
            return False

        # task is up-to-date
        return True

    def get_hash_path(self):
        """Return the path of the hash file

        Returns:
            str: relative path of the hash file
        """
        return os.path.join("hashes", "task_" + self.name + ".hash")

    def read_hash_file(self):
        """Read the hash stored in the hash file

        Returns:
            str: The hash as string
        """
        with open(self.get_hash_path()) as fh:
            return fh.read()

    def copy_input_files(self):
        """Copy all input files to the working directory
        """
        wd = self.get_working_dir()
        for input_file in self.input_files:
            dst = os.path.join(wd, os.path.basename(input_file))
            if os.path.exists(dst):
                log.debug("Remove existing copy of input file \"%s\"",
                          input_file)
                os.remove(dst)
            log.info("Copy input file: %s -> %s", input_file, dst)
            shutil.copy(input_file, dst)

    def create_symlinks(self):
        """Create the specified symlinks
        """
        for symlink in self.symlinks:
            dir = os.path.dirname(symlink[1])
            if not os.path.exists(dir):
                log.info("Create directory: %s", dir)
                os.makedirs(dir)

            relative_symlink(
                os.path.join(self.get_working_dir(), symlink[0]), symlink[1])

    def remove_symlinks(self):
        """Remove the specified symlinks
        """
        for symlink in self.symlinks:
            unlink_if_present(symlink[1])

    def execute(self):
        """Execute the task

        This function has to be implemented in the inheriting classes and
        should do the actual work of the task. This function is executed from
        within the working directory after all input files are copied there.

        Raises:
            NotImplementedError: if this method was not implemented in the
                subclass
        """
        raise NotImplementedError(
            "\"execute\"-method for this task is not implemented")

    def run(self):
        """Run the task

        First the state of the task is checked. If it is up-to-date this method
        returns immediately. Otherwise a clean working directory is created and
        all input files are copied. Then, the task is executed from within the
        working directory. Finally, all symlinks are created and the hash file
        is written.
        """
        cwd = os.getcwd()

        self.show_header()

        if self.is_up_to_date():
            log.info("up-to-date, skip")
            return

        working_dir = self.create_working_dir()
        self.copy_input_files()
        os.chdir(working_dir)
        self.execute()
        os.chdir(cwd)

        self.create_symlinks()
        self.write_hash_file()

        self.remove_working_copies()

    def write_hash_file(self):
        """Write the hash to the hash file

        Hash this instance and write the result to the corresponding hash file.
        """
        with open(self.get_hash_path(), "w") as fh:
            fh.write(self.hash())

    def get_working_dir(self):
        """Get the path of the working directory

        Returns:
            str: relative path to the working directory
        """
        return "task_" + self.name

    def create_working_dir(self):
        """Create the working directory

        Create an empty working directory, this erases any possibly present
        results.
        """
        working_dir = self.get_working_dir()
        if os.path.exists(working_dir):
            log.info("Remove old temporary dir: %s", working_dir)
            shutil.rmtree(working_dir)
        log.info("Create temporary dir: %s", working_dir)
        os.mkdir(working_dir)

        return working_dir

    def remove_working_copies(self):
        """Remove copies of input files
        """
        for input_file in self.input_files:
            path = os.path.join(self.get_working_dir(), os.path.basename(input_file))
            if os.path.exists(path):
                log.debug("Remove working copy: %s", path)
                os.remove(path)
            else:
                log.debug("Working copy of \"%s\" is not present", input_file)

    def clean(self):
        self.remove_symlinks()

        working_dir = self.get_working_dir()
        if os.path.exists(working_dir):
            log.info("Remove working_dir dir: %s", working_dir)
            shutil.rmtree(working_dir)

    def show_header(self):
        """Visually mark the beginning of the task within the info log
        """
        print("")
        log.draw_box("TASK: {}".format(self.name))

    def __str__(self):
        """Represent the task as a string
        """
        return str(self.__dict__)
