import os

import mlxtk.log as log
import mlxtk.operator
import mlxtk.wavefunction


class Project():
    """ML-MCTDH(X) project

    Attributes:
        operators (dict): dictionary containing operator creation functions
            index by names
        root_dir (str): root path for all files created during execution
    """

    def __init__(self, root_dir="."):
        self.operators = {}
        self.wavefunctions = {}
        self.root_dir = root_dir
        self.tasks = []

    def add_operator(self, name, func):
        """Add a new operator to the project

        Args:
            name (str): name of the operator
            func: parameterless function which returns the operator
        """
        self.operators[name] = func

    def add_wavefunction(self, name, func):
        self.wavefunctions[name] = func

    def add_task(self, task):
        self.tasks.append(task)

    def _write_operators(self):
        retval = False

        if not os.path.exists("operators"):
            log.info("Create operator path: %s", "operators")
            os.mkdir("operators")
        operator_updated = {}
        for name in self.operators:
            log.info("Generate operator: %s", name)
            op = self.operators[name]()

            path = os.path.join("operators", name + ".op")
            log.info("Write operator: %s -> %s", name, path)
            updated = mlxtk.operator.write_operator(op, path)
            operator_updated[name] = updated
            retval = (retval or updated)
            if updated:
                log.info("Updated operator file: %s", path)
            else:
                log.info("Operator \"%s\" is up-to-date, skip", name)

        return retval

    def _write_wavefunctions(self):
        retval = False

        if not os.path.exists("wavefunctions"):
            log.info("Create wavefunction path: %s", "wavefunctions")
            os.mkdir("wavefunctions")
        wavefunction_updated = {}
        for name in self.wavefunctions:
            log.info("Generate wave function: %s", name)
            wfn = self.wavefunctions[name]()

            path = os.path.join("wavefunctions", name + ".wfn")
            log.info("Write wave function: %s -> %s", name, path)
            updated = mlxtk.wavefunction.write_wavefunction(wfn, path)
            wavefunction_updated[name] = updated
            retval = (retval or updated)
            if updated:
                log.info("Updated wave function file: %s", path)
            else:
                log.info("Wave function \"%s\" is up-to-date, skip", name)

        return retval

    def is_up_to_date(self):
        if self._write_operators():
            return False

        if self._write_wavefunctions():
            return False

        for task in tasks:
            if not task.is_up_to_date():
                return False

        return True

    def run(self):

        log.info("Start project")
        if not os.path.exists(self.root_dir):
            log.info("Create root path: %s)", self.root_dir)
            os.mkdir(self.root_dir)

        cwd = os.getcwd()
        os.chdir(self.root_dir)

        self._write_operators()
        self._write_wavefunctions()

        if not os.path.exists("hashes"):
            log.info("Create task hashes path: %s", "hashes")
            os.mkdir("hashes")

        for task in self.tasks:
            task.root_dir = self.root_dir
            task.run()
            task.update_project(self)

        os.chdir(cwd)

    def __str__(self):
        return str(self.__dict__)
